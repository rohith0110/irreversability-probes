# Wait for the main ScienceWorld labelling process to exit, then run the thermal pass at
# below-normal priority (the laptop is shared with other work).
param([int]$WaitPid)
$ErrorActionPreference = "Stop"
Set-Location C:\ADNN
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
$env:PYTHONIOENCODING = "utf8"
$p = Start-Process -FilePath C:\ADNN\.venv\Scripts\python.exe -PassThru -WindowStyle Hidden `
    -ArgumentList '-m', 'irrev.sw_label', '--thermal', '--seed', '1', '--out', 'data/raw/scienceworld_thermal.jsonl', '--workers', '8', '--variations-per-task', '5', '--prefixes', '3', '--actions', '4' `
    -WorkingDirectory C:\ADNN -RedirectStandardOutput C:\ADNN\logs\sw_thermal.log -RedirectStandardError C:\ADNN\logs\sw_thermal.err
Start-Sleep -Seconds 60
Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='java.exe'" |
    Where-Object { $_.CommandLine -like '*sw_label*' -or $_.CommandLine -like '*scienceworld.jar*' -or ($_.CommandLine -like '*multiprocessing.spawn*' -and $_.CommandLine -like '*Python311*') } |
    ForEach-Object { try { (Get-Process -Id $_.ProcessId).PriorityClass = 'BelowNormal' } catch {} }
$p.WaitForExit()
