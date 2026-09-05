using System.Collections.Concurrent;
using WatchAbsensi.Application;
using WatchAbsensi.Domain;

namespace WatchAbsensi.Infrastructure;

public sealed class InMemoryAttendanceStore : IAttendanceStore
{
    private readonly ConcurrentDictionary<string, Employee> _employees = new();
    private readonly ConcurrentDictionary<string, WorkShift> _shifts = new();
    private readonly ConcurrentBag<AttendanceRecord> _records = new();
    private readonly ConcurrentBag<SpoofLog> _spoofLogs = new();

    public InMemoryAttendanceStore()
    {
        SeedDefaultShifts();
        SeedDefaultEmployees();
        SeedInitialAttendance();
    }

    public IReadOnlyCollection<Employee> Employees => _employees.Values.ToArray();
    public IReadOnlyCollection<WorkShift> Shifts => _shifts.Values.ToArray();
    public IReadOnlyCollection<AttendanceRecord> Records => _records.ToArray();
    public IReadOnlyCollection<SpoofLog> SpoofLogs => _spoofLogs.ToArray();

    public Employee? GetEmployeeById(string id) =>
        _employees.TryGetValue(id, out var emp) ? emp : null;

    public WorkShift? GetShiftById(string id) =>
        _shifts.TryGetValue(id, out var shift) ? shift : null;

    public AttendanceRecord AddRecord(AttendanceRecord record)
    {
        _records.Add(record);
        return record;
    }

    public SpoofLog AddSpoofLog(SpoofLog log)
    {
        _spoofLogs.Add(log);
        return log;
    }

    public void AddEmployee(Employee employee)
    {
        _employees[employee.Id] = employee;
    }

    public void UpdateEmployeeEmbedding(string id, float[]? embedding)
    {
        if (_employees.TryGetValue(id, out var emp))
        {
            _employees[id] = emp with { FaceEmbedding = embedding, FaceEmbeddings = embedding is null ? null : new[] { embedding } };
        }
    }

    public void UpdateEmployeeEmbeddings(string id, float[][]? embeddings)
    {
        if (_employees.TryGetValue(id, out var emp))
        {
            var cleaned = embeddings?.Where(v => v is { Length: 512 }).ToArray();
            _employees[id] = emp with
            {
                FaceEmbeddings = cleaned is { Length: > 0 } ? cleaned : null,
                FaceEmbedding = cleaned is { Length: > 0 } ? cleaned[0] : null
            };
        }
    }

    public void ResetAllEmbeddings()
    {
        foreach (var key in _employees.Keys)
        {
            if (_employees.TryGetValue(key, out var emp))
            {
                _employees[key] = emp with { FaceEmbedding = null, FaceEmbeddings = null };
            }
        }
    }

    private void SeedDefaultShifts()
    {
        var morning = new WorkShift(
            Id: "SHIFT-MORNING",
            Name: "Standard Morning Shift",
            StartTime: new TimeOnly(8, 0),
            EndTime: new TimeOnly(17, 0),
            GracePeriodMinutes: 15,
            LateThresholdMinutes: 60
        );

        var afternoon = new WorkShift(
            Id: "SHIFT-AFTERNOON",
            Name: "Kiosk Afternoon Shift",
            StartTime: new TimeOnly(13, 0),
            EndTime: new TimeOnly(21, 0),
            GracePeriodMinutes: 15,
            LateThresholdMinutes: 60
        );

        _shifts[morning.Id] = morning;
        _shifts[afternoon.Id] = afternoon;
    }

    private void SeedDefaultEmployees()
    {
        // 8 Personnel - Starting from Willy A. with alphabetical surnames (A, B, C, D, E, F, G, H)
        // Initially ALL ZERO (FaceEmbedding = null), enrolled one-by-one.
        var employees = new[]
        {
            new Employee(
                Id: "EMP-001",
                Nip: "20260901-001",
                FullName: "Willy Arlando",
                Department: "Technology & Laboratories",
                Role: "System Lead & Primary Tester",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-100)
            ),
            new Employee(
                Id: "EMP-002",
                Nip: "20260901-002",
                FullName: "Kenny Baskoro",
                Department: "Faculty / Teaching Staff",
                Role: "Mathematics & Science Instructor",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1570295999919-56ceb5ecca61?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-90)
            ),
            new Employee(
                Id: "EMP-003",
                Nip: "20260901-003",
                FullName: "Jessy Chandra",
                Department: "General Administration",
                Role: "Head Administrative Officer",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-80)
            ),
            new Employee(
                Id: "EMP-004",
                Nip: "20260901-004",
                FullName: "Brendy Darmawan",
                Department: "Information Technology",
                Role: "Infrastructure & Network Engineer",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-70)
            ),
            new Employee(
                Id: "EMP-005",
                Nip: "20260901-005",
                FullName: "Elly Efendi",
                Department: "Faculty / Teaching Staff",
                Role: "Language & Communications Lecturer",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-60)
            ),
            new Employee(
                Id: "EMP-006",
                Nip: "20260901-006",
                FullName: "Niggy Fahrezi",
                Department: "Human Resources / HR",
                Role: "Personnel Operations Specialist",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-50)
            ),
            new Employee(
                Id: "EMP-007",
                Nip: "20260901-007",
                FullName: "Micky Gunawan",
                Department: "Operations & Facilities",
                Role: "Facilities & Logistics Coordinator",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-40)
            ),
            new Employee(
                Id: "EMP-008",
                Nip: "20260901-008",
                FullName: "Jibby Hartono",
                Department: "Finance & Logistics",
                Role: "Financial Administration Officer",
                ShiftId: "SHIFT-MORNING",
                FaceEmbedding: null,
                Active: true,
                AvatarUrl: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80",
                CreatedAt: DateTimeOffset.UtcNow.AddDays(-30)
            )
        };

        foreach (var emp in employees)
        {
            _employees[emp.Id] = emp;
        }
    }

    private void SeedInitialAttendance()
    {
        // Awalnya dari nol, seluruh data presensi baru akan terisi saat wajah didaftarkan dan absensi dilakukan.
    }

    private static float[] GenerateSeededEmbedding(int seed)
    {
        var random = new Random(seed * 42);
        var vector = new float[512];
        float sumSq = 0f;
        for (int i = 0; i < 512; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            sumSq += vector[i] * vector[i];
        }
        float norm = MathF.Sqrt(sumSq);
        for (int i = 0; i < 512; i++)
        {
            vector[i] /= norm;
        }
        return vector;
    }
}

public sealed class InMemoryAntiDoubleTapService : IAntiDoubleTapService
{
    private readonly ConcurrentDictionary<string, DateTimeOffset> _lastAttempts = new();
    private readonly TimeSpan _debounceWindow;

    public InMemoryAntiDoubleTapService(TimeSpan? debounceWindow = null)
    {
        _debounceWindow = debounceWindow ?? TimeSpan.FromSeconds(60);
    }

    public bool TryAcquire(string employeeId, out TimeSpan retryAfter)
    {
        var now = DateTimeOffset.UtcNow;
        if (_lastAttempts.TryGetValue(employeeId, out var lastTime))
        {
            var elapsed = now - lastTime;
            if (elapsed < _debounceWindow)
            {
                retryAfter = _debounceWindow - elapsed;
                return false;
            }
        }

        _lastAttempts[employeeId] = now;
        retryAfter = TimeSpan.Zero;
        return true;
    }

    public void Reset(string employeeId)
    {
        _lastAttempts.TryRemove(employeeId, out _);
    }
}
