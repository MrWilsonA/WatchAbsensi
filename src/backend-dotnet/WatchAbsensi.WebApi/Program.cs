using Microsoft.AspNetCore.SignalR; using WatchAbsensi.Application; using WatchAbsensi.Domain; using WatchAbsensi.Infrastructure;
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton<IAttendanceStore, InMemoryAttendanceStore>();
builder.Services.AddSignalR(); builder.Services.AddHealthChecks(); builder.Services.AddCors(o=>o.AddDefaultPolicy(p=>p.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));
var app=builder.Build(); app.UseCors(); app.MapHealthChecks("/health");
app.MapGet("/api/v1/attendance", (IAttendanceStore s, int? limit) => Results.Ok(s.All.OrderByDescending(x=>x.RecordedAt).Take(Math.Clamp(limit??50,1,500))));
app.MapPost("/api/v1/attendance", async (AttendanceInput input, IAttendanceStore s, IHubContext<AttendanceHub> hub) => { if(input.Confidence<.75||input.Liveness<.8) return Results.BadRequest(new {success=false,reason="Biometric threshold not met"}); var record=s.Add(new(Guid.NewGuid(),input.EmployeeId,input.Mode,DateTimeOffset.UtcNow,input.Confidence,input.Liveness)); await hub.Clients.All.SendAsync("attendanceRecorded",record); return Results.Created($"/api/v1/attendance/{record.Id}",new {success=true,record}); });
app.MapHub<AttendanceHub>("/hubs/attendance"); app.Run();
public record AttendanceInput(string EmployeeId, AttendanceMode Mode=AttendanceMode.CheckIn, double Confidence=.98, double Liveness=.99);
public sealed class AttendanceHub:Hub { }
