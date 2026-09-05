using WatchAbsensi.Domain;

namespace WatchAbsensi.Application;

public interface IAttendanceStore
{
    IReadOnlyCollection<Employee> Employees { get; }
    IReadOnlyCollection<WorkShift> Shifts { get; }
    IReadOnlyCollection<AttendanceRecord> Records { get; }
    IReadOnlyCollection<SpoofLog> SpoofLogs { get; }

    Employee? GetEmployeeById(string id);
    WorkShift? GetShiftById(string id);
    AttendanceRecord AddRecord(AttendanceRecord record);
    SpoofLog AddSpoofLog(SpoofLog log);
    void AddEmployee(Employee employee);
    void UpdateEmployeeEmbedding(string id, float[]? embedding);
    void UpdateEmployeeEmbeddings(string id, float[][]? embeddings);
    void ResetAllEmbeddings();
}

public interface IAntiDoubleTapService
{
    bool TryAcquire(string employeeId, out TimeSpan retryAfter);
    void Reset(string employeeId);
}

public interface IBiometricService
{
    double CalculateCosineSimilarity(float[] vectorA, float[] vectorB);
    (Employee? Employee, double Similarity) FindBestMatch(float[] queryEmbedding, double threshold = 0.65);
}

public record AttendanceSubmission(
    string EmployeeId,
    AttendanceMode Mode = AttendanceMode.CheckIn,
    double Confidence = 0.98,
    double Liveness = 0.99,
    string DeviceId = "KIOSK-MAIN-01",
    string? ShiftId = null,
    float[]? FaceEmbedding = null
);

public record AttendanceResult(
    bool Success,
    string Status,
    string Message,
    AttendanceRecord? Record = null,
    TimeSpan? RetryAfter = null,
    SpoofLog? SpoofDetails = null
);

public class BiometricService : IBiometricService
{
    private readonly IAttendanceStore _store;

    public BiometricService(IAttendanceStore store)
    {
        _store = store;
    }

    public double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length == 0 || vectorB.Length == 0 || vectorA.Length != vectorB.Length)
            return 0.0;

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        if (normA <= 0.0 || normB <= 0.0)
            return 0.0;

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public (Employee? Employee, double Similarity) FindBestMatch(float[] queryEmbedding, double threshold = 0.65)
    {
        Employee? bestEmployee = null;
        double bestSimilarity = 0.0;

        foreach (var emp in _store.Employees)
        {
            if (!emp.Active || emp.FaceEmbedding == null)
                continue;

            var candidates = emp.FaceEmbeddings?.Length > 0 ? emp.FaceEmbeddings : new[] { emp.FaceEmbedding };
            foreach (var candidate in candidates)
            {
                if (candidate is null) continue;
                double sim = CalculateCosineSimilarity(queryEmbedding, candidate);
                if (sim > bestSimilarity)
                {
                    bestSimilarity = sim;
                    bestEmployee = emp;
                }
            }
        }

        if (bestSimilarity >= threshold)
        {
            return (bestEmployee, bestSimilarity);
        }

        return (null, bestSimilarity);
    }
}

public class PunctualityEngine
{
    public static (AttendanceStatus Status, int LateMinutes) Evaluate(
        AttendanceMode mode,
        DateTimeOffset recordedAt,
        WorkShift shift)
    {
        var localTime = TimeOnly.FromDateTime(recordedAt.LocalDateTime);

        if (mode == AttendanceMode.CheckIn)
        {
            var graceThreshold = shift.StartTime.AddMinutes(shift.GracePeriodMinutes);
            if (localTime <= graceThreshold)
            {
                return (AttendanceStatus.OnTime, 0);
            }

            int diffMinutes = (int)(localTime.ToTimeSpan() - shift.StartTime.ToTimeSpan()).TotalMinutes;
            return (AttendanceStatus.Late, Math.Max(1, diffMinutes));
        }
        else // CheckOut
        {
            if (localTime < shift.EndTime.AddMinutes(-10))
            {
                int earlyMinutes = (int)(shift.EndTime.ToTimeSpan() - localTime.ToTimeSpan()).TotalMinutes;
                return (AttendanceStatus.EarlyDeparture, Math.Max(1, earlyMinutes));
            }

            if (localTime >= shift.EndTime.AddHours(1))
            {
                return (AttendanceStatus.Overtime, 0);
            }

            return (AttendanceStatus.Normal, 0);
        }
    }
}
