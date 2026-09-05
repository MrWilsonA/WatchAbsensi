FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /source
COPY . .
RUN dotnet restore WatchAbsensi.WebApi/WatchAbsensi.WebApi.csproj
RUN dotnet publish WatchAbsensi.WebApi/WatchAbsensi.WebApi.csproj -c Release -o /app/publish --no-restore
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_HTTP_PORTS=8080
EXPOSE 8080
ENTRYPOINT ["dotnet","WatchAbsensi.WebApi.dll"]
