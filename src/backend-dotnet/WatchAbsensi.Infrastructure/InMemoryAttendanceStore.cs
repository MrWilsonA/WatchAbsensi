using WatchAbsensi.Application; using WatchAbsensi.Domain; using System.Collections.Concurrent;
namespace WatchAbsensi.Infrastructure;
public sealed class InMemoryAttendanceStore : IAttendanceStore { private readonly ConcurrentBag<AttendanceRecord> _records = new(); public IReadOnlyCollection<AttendanceRecord> All => _records.ToArray(); public AttendanceRecord Add(AttendanceRecord record){_records.Add(record); return record;} }
