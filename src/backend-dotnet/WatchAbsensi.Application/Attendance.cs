using WatchAbsensi.Domain;
namespace WatchAbsensi.Application;
public interface IAttendanceStore { IReadOnlyCollection<AttendanceRecord> All { get; } AttendanceRecord Add(AttendanceRecord record); }
