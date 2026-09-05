namespace WatchAbsensi.Domain;
public enum AttendanceMode { CheckIn, CheckOut }
public record Employee(string Id, string Name, string Department, bool Active = true);
public record AttendanceRecord(Guid Id, string EmployeeId, AttendanceMode Mode, DateTimeOffset RecordedAt, double Confidence, double Liveness);
