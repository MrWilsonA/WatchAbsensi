using Microsoft.AspNetCore.SignalR;
using WatchAbsensi.Application;
using WatchAbsensi.Domain;
using WatchAbsensi.Infrastructure;

var builder = WebApplication.CreateBuilder(args);

// Dependency Injection
builder.Services.AddSingleton<IAttendanceStore, InMemoryAttendanceStore>();
builder.Services.AddSingleton<IAntiDoubleTapService, InMemoryAntiDoubleTapService>();
builder.Services.AddSingleton<IBiometricService, BiometricService>();

builder.Services.AddSignalR();
builder.Services.AddHealthChecks();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.SetIsOriginAllowed(_ => true)
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

var app = builder.Build();

app.UseCors();
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "WatchAbsensi .NET API v1");
    c.RoutePrefix = "swagger";
});

app.MapHealthChecks("/health");

// --- SignalR Hub ---
app.MapHub<AttendanceHub>("/hubs/attendance");

// --- Employee Endpoints ---
app.MapGet("/api/v1/employees", (IAttendanceStore store, string? department) =>
{
    var employees = store.Employees.AsEnumerable();
    if (!string.IsNullOrWhiteSpace(department))
    {
        employees = employees.Where(e => e.Department.Equals(department, StringComparison.OrdinalIgnoreCase));
    }
    return Results.Ok(employees.OrderBy(e => e.FullName));
})
.WithName("GetEmployees")
.WithTags("Employees");

app.MapGet("/api/v1/employees/{id}", (string id, IAttendanceStore store) =>
{
    var employee = store.GetEmployeeById(id);
    return employee is not null ? Results.Ok(employee) : Results.NotFound(new { message = $"Employee '{id}' not found" });
})
.WithName("GetEmployeeById")
.WithTags("Employees");

app.MapPost("/api/v1/employees", (Employee employee, IAttendanceStore store) =>
{
    if (string.IsNullOrWhiteSpace(employee.Id) || string.IsNullOrWhiteSpace(employee.FullName))
    {
        return Results.BadRequest(new { message = "Id and FullName are required." });
    }

    var record = employee with { CreatedAt = DateTimeOffset.UtcNow };
    store.AddEmployee(record);
    return Results.Created($"/api/v1/employees/{record.Id}", record);
})
.WithName("CreateEmployee")
.WithTags("Employees");

// --- Employee Face Enrollment & Management Endpoints ---
app.MapPost("/api/v1/employees/{id}/enroll", (string id, EnrollFaceRequest req, IAttendanceStore store) =>
{
    var employee = store.GetEmployeeById(id);
    if (employee is null)
        return Results.NotFound(new { success = false, message = $"Personnel with ID '{id}' was not found." });

    var embeddings = req.Embeddings is { Length: > 0 } ? req.Embeddings : (req.Embedding is { Length: 512 } ? new[] { req.Embedding } : null);
    if (embeddings is null || embeddings.Any(v => v is null || v.Length != 512))
        return Results.BadRequest(new { success = false, message = "Enrollment requires 1-5 biometric vectors, each 512 dimensions." });

    if (embeddings.Length > 5)
        return Results.BadRequest(new { success = false, message = "Maximum 5 face samples per employee." });

    const double duplicateThreshold = 0.92;
    foreach (var other in store.Employees.Where(e => e.Id != id && e.Active))
    {
        var registered = other.FaceEmbeddings?.Length > 0 ? other.FaceEmbeddings : (other.FaceEmbedding is null ? Array.Empty<float[]>() : new[] { other.FaceEmbedding });
        foreach (var incoming in embeddings)
        foreach (var existing in registered)
        {
            var similarity = new BiometricService(store).CalculateCosineSimilarity(incoming, existing);
            if (similarity >= duplicateThreshold)
            {
                return Results.Conflict(new { success = false, status = "already_registered", message = $"Wajah ini sudah terdaftar sebagai {other.FullName}.", existingEmployeeId = other.Id, existingEmployeeName = other.FullName, similarity = Math.Round(similarity, 4) });
            }
        }
    }

    store.UpdateEmployeeEmbeddings(id, embeddings);
    return Results.Ok(new { success = true, status = "enrolled", message = $"Face template successfully enrolled for personnel {employee.FullName}.", employeeId = id, samples = embeddings.Length });
})
.WithName("EnrollEmployeeFace")
.WithTags("Employees");

app.MapPost("/api/v1/employees/reset-embeddings", (IAttendanceStore store, IAntiDoubleTapService antiDoubleTap) =>
{
    store.ResetAllEmbeddings();
    foreach (var emp in store.Employees)
    {
        antiDoubleTap.Reset(emp.Id);
    }
    return Results.Ok(new { success = true, message = "All personnel face templates have been reset to zero." });
})
.WithName("ResetAllEmployeeEmbeddings")
.WithTags("Employees");

// --- Shift Endpoints ---
app.MapGet("/api/v1/shifts", (IAttendanceStore store) => Results.Ok(store.Shifts))
.WithName("GetShifts")
.WithTags("Shifts");

// --- Attendance Endpoints ---
app.MapGet("/api/v1/attendance", (IAttendanceStore store, int? limit, string? mode, string? department) =>
{
    var query = store.Records.AsEnumerable();

    if (!string.IsNullOrWhiteSpace(mode) && Enum.TryParse<AttendanceMode>(mode, true, out var parsedMode))
    {
        query = query.Where(r => r.Mode == parsedMode);
    }

    if (!string.IsNullOrWhiteSpace(department))
    {
        query = query.Where(r => r.Department.Equals(department, StringComparison.OrdinalIgnoreCase));
    }

    var result = query
        .OrderByDescending(r => r.RecordedAt)
        .Take(Math.Clamp(limit ?? 50, 1, 500))
        .ToList();

    return Results.Ok(result);
})
.WithName("GetAttendanceRecords")
.WithTags("Attendance");

app.MapPost("/api/v1/attendance", async (
    AttendanceSubmission submission,
    IAttendanceStore store,
    IAntiDoubleTapService antiDoubleTap,
    IBiometricService biometricService,
    IHubContext<AttendanceHub> hub) =>
{
    // 1. Biometric & Liveness Verification Check
    if (submission.Liveness < 0.80 || submission.Confidence < 0.75)
    {
        var spoofReason = submission.Liveness < 0.80 ? SpoofReason.TextureAnomaly : SpoofReason.LowConfidence;
        var spoofLog = new SpoofLog(
            Id: Guid.NewGuid(),
            RecordedAt: DateTimeOffset.UtcNow,
            DeviceId: submission.DeviceId,
            Liveness: submission.Liveness,
            Confidence: submission.Confidence,
            Reason: spoofReason,
            Details: $"Verification rejected. Liveness: {submission.Liveness:P1}, Confidence: {submission.Confidence:P1}",
            CandidateEmployeeId: submission.EmployeeId
        );

        store.AddSpoofLog(spoofLog);
        await hub.Clients.All.SendAsync("spoofDetected", spoofLog);

        return Results.BadRequest(new AttendanceResult(
            Success: false,
            Status: "spoof_rejected",
            Message: "Biometric verification failed: Liveness or confidence score below safety threshold.",
            SpoofDetails: spoofLog
        ));
    }

    // 2. Anti-Double Tap / Debounce Enforcement
    if (!antiDoubleTap.TryAcquire(submission.EmployeeId, out var retryAfter))
    {
        var debouncePayload = new
        {
            employeeId = submission.EmployeeId,
            message = $"Transaction throttled to prevent duplicate scans (Anti-Double-Tap). Please retry in {Math.Ceiling(retryAfter.TotalSeconds)} seconds.",
            retryAfterSeconds = Math.Ceiling(retryAfter.TotalSeconds)
        };

        await hub.Clients.All.SendAsync("doubleTapPrevented", debouncePayload);

        return Results.Conflict(new AttendanceResult(
            Success: false,
            Status: "double_tap_prevented",
            Message: debouncePayload.message,
            RetryAfter: retryAfter
        ));
    }

    // 3. Resolve Employee
    var employee = store.GetEmployeeById(submission.EmployeeId);
    if (employee is null)
    {
        // Try fallback match by biometric embedding if provided
        if (submission.FaceEmbedding != null && submission.FaceEmbedding.Length == 512)
        {
            var match = biometricService.FindBestMatch(submission.FaceEmbedding);
            employee = match.Employee;
        }

        if (employee is null)
        {
            antiDoubleTap.Reset(submission.EmployeeId);
            return Results.NotFound(new AttendanceResult(
                Success: false,
                Status: "employee_not_found",
                Message: $"Personnel with ID '{submission.EmployeeId}' was not found in database."
            ));
        }
    }

    // 4. Determine Shift & Evaluate Punctuality
    var shiftId = submission.ShiftId ?? employee.ShiftId;
    var shift = store.GetShiftById(shiftId) ?? store.Shifts.First();
    var recordedAt = DateTimeOffset.UtcNow;
    var (status, lateMinutes) = PunctualityEngine.Evaluate(submission.Mode, recordedAt, shift);

    // 5. Create and Store Attendance Record
    var record = new AttendanceRecord(
        Id: Guid.NewGuid(),
        EmployeeId: employee.Id,
        EmployeeName: employee.FullName,
        Department: employee.Department,
        Mode: submission.Mode,
        Status: status,
        RecordedAt: recordedAt,
        Confidence: submission.Confidence,
        Liveness: submission.Liveness,
        ShiftId: shift.Id,
        LateMinutes: lateMinutes,
        DeviceId: submission.DeviceId
    );

    store.AddRecord(record);

    // 6. Broadcast via SignalR to all connected kiosks/dashboards
    await hub.Clients.All.SendAsync("attendanceRecorded", record);

    var statusDesc = status switch
    {
        AttendanceStatus.OnTime => "On-Time",
        AttendanceStatus.Late => $"Late by {lateMinutes} minutes",
        AttendanceStatus.EarlyDeparture => "Early Departure",
        AttendanceStatus.Overtime => "Overtime Logged",
        _ => "Standard"
    };

    return Results.Created($"/api/v1/attendance/{record.Id}", new AttendanceResult(
        Success: true,
        Status: "recorded",
        Message: $"Attendance {record.Mode} recorded: {employee.FullName} ({statusDesc})",
        Record: record
    ));
})
.WithName("RecordAttendance")
.WithTags("Attendance");

// --- Biometric Vector Matcher Endpoint ---
app.MapPost("/api/v1/attendance/verify", (BiometricVerifyRequest req, IBiometricService biometricService) =>
{
    if (req.Embedding == null || req.Embedding.Length != 512)
    {
        return Results.BadRequest(new { message = "Vector embedding must be 512 dimensions." });
    }

    var (matchedEmployee, similarity) = biometricService.FindBestMatch(req.Embedding, req.Threshold ?? 0.75);
    if (matchedEmployee is null)
    {
        return Results.Ok(new
        {
            matched = false,
            similarity = Math.Round(similarity, 4),
            message = "Face does not match any enrolled personnel."
        });
    }

    return Results.Ok(new
    {
        matched = true,
        similarity = Math.Round(similarity, 4),
        employee = matchedEmployee
    });
})
.WithName("VerifyBiometrics")
.WithTags("Attendance");

// --- Auto-Scan Attendance (Face Identification & Instant Clock-in) ---
app.MapPost("/api/v1/attendance/auto-scan", async (
    AutoScanSubmission scan,
    IAttendanceStore store,
    IAntiDoubleTapService antiDoubleTap,
    IBiometricService biometricService,
    IHubContext<AttendanceHub> hub) =>
{
    if (scan.Embedding == null || scan.Embedding.Length != 512)
        return Results.BadRequest(new { success = false, status = "invalid_vector", message = "Biometric vector embedding must be 512 dimensions." });

    // 1. Biometric threshold check
    if (scan.Liveness < 0.80 || scan.Confidence < 0.75)
    {
        var spoofLog = new SpoofLog(
            Id: Guid.NewGuid(),
            RecordedAt: DateTimeOffset.UtcNow,
            DeviceId: scan.DeviceId,
            Liveness: scan.Liveness,
            Confidence: scan.Confidence,
            Reason: scan.Liveness < 0.80 ? SpoofReason.TextureAnomaly : SpoofReason.LowConfidence,
            Details: "Liveness score below safety threshold (Spoofing attempt detected)",
            CandidateEmployeeId: null
        );
        store.AddSpoofLog(spoofLog);
        await hub.Clients.All.SendAsync("spoofDetected", spoofLog);
        return Results.BadRequest(new AttendanceResult(false, "spoof_rejected", "Biometric verification failed: Liveness anomaly detected.", null, null, spoofLog));
    }

    // 2. Identify person from enrolled embeddings
    var (matchedEmployee, similarity) = biometricService.FindBestMatch(scan.Embedding, scan.Threshold ?? 0.75);
    if (matchedEmployee == null)
    {
        return Results.NotFound(new {
            success = false,
            status = "unrecognized_face",
            similarity = Math.Round(similarity, 4),
            message = "Unrecognized face / Not enrolled in the directory. Please enroll face template first."
        });
    }

    // 3. Check anti double-tap
    if (!antiDoubleTap.TryAcquire(matchedEmployee.Id, out var retryAfter))
    {
        var debouncePayload = new
        {
            employeeId = matchedEmployee.Id,
            message = $"Anti-Double-Tap throttled: {matchedEmployee.FullName} already recorded. Please wait {Math.Ceiling(retryAfter.TotalSeconds)} seconds.",
            retryAfterSeconds = Math.Ceiling(retryAfter.TotalSeconds)
        };
        await hub.Clients.All.SendAsync("doubleTapPrevented", debouncePayload);
        return Results.Conflict(new AttendanceResult(false, "double_tap_prevented", debouncePayload.message, null, retryAfter));
    }

    // 4. Determine shift & calculate punctuality
    var shiftId = scan.ShiftId ?? matchedEmployee.ShiftId;
    var shift = store.GetShiftById(shiftId) ?? store.Shifts.First();
    var recordedAt = DateTimeOffset.UtcNow;
    var (status, lateMinutes) = PunctualityEngine.Evaluate(scan.Mode, recordedAt, shift);

    var record = new AttendanceRecord(
        Id: Guid.NewGuid(),
        EmployeeId: matchedEmployee.Id,
        EmployeeName: matchedEmployee.FullName,
        Department: matchedEmployee.Department,
        Mode: scan.Mode,
        Status: status,
        RecordedAt: recordedAt,
        Confidence: scan.Confidence,
        Liveness: scan.Liveness,
        ShiftId: shift.Id,
        LateMinutes: lateMinutes,
        DeviceId: scan.DeviceId
    );

    store.AddRecord(record);
    await hub.Clients.All.SendAsync("attendanceRecorded", record);

    var statusDesc = status == AttendanceStatus.OnTime ? "On-Time" : $"Late by {lateMinutes} minutes";
    var modeDesc = scan.Mode == AttendanceMode.CheckIn ? "Check-In" : "Check-Out";
    return Results.Created($"/api/v1/attendance/{record.Id}", new AttendanceResult(
        Success: true,
        Status: "recorded",
        Message: $"Attendance {modeDesc} recorded: {matchedEmployee.FullName} ({statusDesc}, Match: {similarity:P1})",
        Record: record
    ));
})
.WithName("AutoScanAttendance")
.WithTags("Attendance");

// --- Spoof Incident Endpoint ---
app.MapPost("/api/v1/attendance/spoof", async (SpoofLogSubmission input, IAttendanceStore store, IHubContext<AttendanceHub> hub) =>
{
    var log = new SpoofLog(
        Id: Guid.NewGuid(),
        RecordedAt: DateTimeOffset.UtcNow,
        DeviceId: input.DeviceId ?? "KIOSK-MAIN-01",
        Liveness: input.Liveness,
        Confidence: input.Confidence,
        Reason: input.Reason,
        Details: input.Details ?? "Biometric spoof attempt prevented.",
        CandidateEmployeeId: input.CandidateEmployeeId
    );

    store.AddSpoofLog(log);
    await hub.Clients.All.SendAsync("spoofDetected", log);

    return Results.Ok(new { success = true, log });
})
.WithName("LogSpoofIncident")
.WithTags("Analytics");

// --- Analytics Endpoints ---
app.MapGet("/api/v1/analytics/summary", (IAttendanceStore store) =>
{
    var todayUtc = DateTimeOffset.UtcNow.Date;
    var todayRecords = store.Records
        .Where(r => r.RecordedAt.Date == todayUtc)
        .ToList();

    var presentEmployeeIds = todayRecords
        .Where(r => r.Mode == AttendanceMode.CheckIn)
        .Select(r => r.EmployeeId)
        .Distinct()
        .ToHashSet();

    int totalEmployees = store.Employees.Count;
    int presentCount = presentEmployeeIds.Count;
    int onTimeCount = todayRecords.Count(r => r.Mode == AttendanceMode.CheckIn && r.Status == AttendanceStatus.OnTime);
    int lateCount = todayRecords.Count(r => r.Mode == AttendanceMode.CheckIn && r.Status == AttendanceStatus.Late);
    int absentCount = Math.Max(0, totalEmployees - presentCount);
    double attendanceRate = totalEmployees > 0 ? (double)presentCount / totalEmployees : 0.0;

    int spoofCount = store.SpoofLogs.Count(s => s.RecordedAt.Date == todayUtc);

    var summary = new AttendanceSummary(
        Date: todayUtc.ToString("yyyy-MM-dd"),
        TotalEmployees: totalEmployees,
        PresentCount: presentCount,
        OnTimeCount: onTimeCount,
        LateCount: lateCount,
        AbsentCount: absentCount,
        AttendanceRate: Math.Round(attendanceRate, 3),
        AverageWorkHours: 8.2,
        SpoofAttemptsPrevented: spoofCount
    );

    return Results.Ok(summary);
})
.WithName("GetAnalyticsSummary")
.WithTags("Analytics");

app.MapGet("/api/v1/analytics/hourly", (IAttendanceStore store) =>
{
    var todayUtc = DateTimeOffset.UtcNow.Date;
    var todayRecords = store.Records
        .Where(r => r.RecordedAt.Date == todayUtc)
        .ToList();

    var hourlyList = new List<HourlyTraffic>();
    for (int hour = 6; hour <= 20; hour++)
    {
        int checkIns = todayRecords.Count(r => r.Mode == AttendanceMode.CheckIn && r.RecordedAt.Hour == hour);
        int checkOuts = todayRecords.Count(r => r.Mode == AttendanceMode.CheckOut && r.RecordedAt.Hour == hour);

        hourlyList.Add(new HourlyTraffic(
            Hour: hour,
            Label: $"{hour:D2}:00",
            CheckInCount: checkIns,
            CheckOutCount: checkOuts
        ));
    }

    return Results.Ok(hourlyList);
})
.WithName("GetHourlyTraffic")
.WithTags("Analytics");

app.MapGet("/api/v1/analytics/departments", (IAttendanceStore store) =>
{
    var todayUtc = DateTimeOffset.UtcNow.Date;
    var todayRecords = store.Records
        .Where(r => r.RecordedAt.Date == todayUtc && r.Mode == AttendanceMode.CheckIn)
        .ToList();

    var departments = store.Employees
        .GroupBy(e => e.Department)
        .Select(g =>
        {
            int total = g.Count();
            var checkedIn = todayRecords.Where(r => r.Department == g.Key).ToList();
            int present = checkedIn.Select(r => r.EmployeeId).Distinct().Count();
            int late = checkedIn.Count(r => r.Status == AttendanceStatus.Late);
            double rate = total > 0 ? (double)(present - late) / total : 0.0;

            return new DepartmentAttendance(
                Department: g.Key,
                Total: total,
                Present: present,
                Late: late,
                PunctualityRate: Math.Max(0.0, Math.Round(rate, 2))
            );
        })
        .ToList();

    return Results.Ok(departments);
})
.WithName("GetDepartmentAttendance")
.WithTags("Analytics");

app.MapGet("/api/v1/analytics/spoof-logs", (IAttendanceStore store, int? limit) =>
{
    var logs = store.SpoofLogs
        .OrderByDescending(s => s.RecordedAt)
        .Take(Math.Clamp(limit ?? 50, 1, 200))
        .ToList();

    return Results.Ok(logs);
})
.WithName("GetSpoofLogs")
.WithTags("Analytics");

app.Run();

// --- Hub & DTO Classes ---
public sealed class AttendanceHub : Hub
{
    public async Task BroadcastPresence(string clientName)
    {
        await Clients.Others.SendAsync("clientJoined", clientName);
    }
}

public record BiometricVerifyRequest(float[] Embedding, double? Threshold = 0.75);

public record EnrollFaceRequest(float[]? Embedding = null, float[][]? Embeddings = null);

public record AutoScanSubmission(
    float[] Embedding,
    AttendanceMode Mode = AttendanceMode.CheckIn,
    double Confidence = 0.98,
    double Liveness = 0.99,
    string DeviceId = "KIOSK-01",
    string? ShiftId = null,
    double? Threshold = 0.75
);

public record SpoofLogSubmission(
    double Liveness,
    double Confidence,
    SpoofReason Reason = SpoofReason.TextureAnomaly,
    string? Details = null,
    string? DeviceId = "KIOSK-MAIN-01",
    string? CandidateEmployeeId = null
);
