namespace WatchAbsensi.Domain;

public enum AttendanceMode
{
    CheckIn,
    CheckOut
}

public enum AttendanceStatus
{
    OnTime,
    Late,
    EarlyDeparture,
    Overtime,
    Normal
}

public enum SpoofReason
{
    TextureAnomaly,
    ScreenReflection,
    LowConfidence,
    BlinkMissing,
    PhotoPrintDetected
}

public record WorkShift(
    string Id,
    string Name,
    TimeOnly StartTime,
    TimeOnly EndTime,
    int GracePeriodMinutes = 15,
    int LateThresholdMinutes = 60
);

public record Employee(
    string Id,
    string Nip,
    string FullName,
    string Department,
    string Role,
    string ShiftId = "SHIFT-MORNING",
    float[]? FaceEmbedding = null,
    bool Active = true,
    string? AvatarUrl = null,
    DateTimeOffset? CreatedAt = null,
    float[][]? FaceEmbeddings = null
);

public record AttendanceRecord(
    Guid Id,
    string EmployeeId,
    string EmployeeName,
    string Department,
    AttendanceMode Mode,
    AttendanceStatus Status,
    DateTimeOffset RecordedAt,
    double Confidence,
    double Liveness,
    string ShiftId = "SHIFT-MORNING",
    int LateMinutes = 0,
    string DeviceId = "KIOSK-MAIN-01"
);

public record SpoofLog(
    Guid Id,
    DateTimeOffset RecordedAt,
    string DeviceId,
    double Liveness,
    double Confidence,
    SpoofReason Reason,
    string Details,
    string? CandidateEmployeeId = null
);

public record AttendanceSummary(
    string Date,
    int TotalEmployees,
    int PresentCount,
    int OnTimeCount,
    int LateCount,
    int AbsentCount,
    double AttendanceRate,
    double AverageWorkHours,
    int SpoofAttemptsPrevented
);

public record HourlyTraffic(
    int Hour,
    string Label,
    int CheckInCount,
    int CheckOutCount
);

public record DepartmentAttendance(
    string Department,
    int Total,
    int Present,
    int Late,
    double PunctualityRate
);
