# fetch_occurrences.ps1 - fetch Muntiacus crinifrons occurrences from GBIF API
$taxonKey = 2440945
$out = 'REPO_ROOT\deep-sea-exploration\species-search\data\gbif\occurrences.csv'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

$all = @()
$offset = 0
$limit = 300
$total = $null

while ($true) {
    $url = "https://api.gbif.org/v1/occurrence/search?taxonKey=$taxonKey&hasCoordinate=true&hasGeospatialIssue=false&limit=$limit&offset=$offset"
    try {
        $r = Invoke-RestMethod $url -TimeoutSec 60
    } catch {
        Write-Host "[warn] offset=$offset request failed: $($_.Exception.Message), retry x3"
        $ok = $false
        for ($i = 1; $i -le 3; $i++) {
            Start-Sleep -Seconds 3
            try { $r = Invoke-RestMethod $url -TimeoutSec 60; $ok = $true; break } catch { Write-Host "  retry $i failed" }
        }
        if (-not $ok) { Write-Host "[fail] offset=$offset give up"; break }
    }
    if ($null -eq $total) {
        $total = $r.count
        Write-Host "[info] total = $total"
    }
    if ($null -eq $r.results -or $r.results.Count -eq 0) { break }
    foreach ($rec in $r.results) {
        $lat = $rec.decimalLatitude; $lon = $rec.decimalLongitude
        if ($null -eq $lat -or $null -eq $lon) { continue }
        $unc = $rec.coordinateUncertaintyInMeters
        if ($null -eq $unc) { $unc = 0 }
        $all += [pscustomobject]@{
            gbifID = $rec.key
            lat = [double]$lat
            lon = [double]$lon
            year = if ($null -ne $rec.year) { $rec.year } else { '' }
            basisOfRecord = if ($null -ne $rec.basisOfRecord) { $rec.basisOfRecord } else { '' }
            uncertainty_m = [double]$unc
        }
    }
    $offset += $r.results.Count
    Write-Host "[info] fetched $($all.Count)/$total"
    if ($offset -ge $total -or $r.results.Count -lt $limit) { break }
    Start-Sleep -Milliseconds 400
}

$all | Export-Csv $out -NoTypeInformation -Encoding UTF8
Write-Host "[done] saved $($all.Count) rows -> $out"
$precise = $all | Where-Object { $_.uncertainty_m -le 5000 }
Write-Host "[stat] precise records (unc<=5000m): $($precise.Count)"
if ($all.Count -gt 0) {
    $lats = $all | ForEach-Object { $_.lat }
    $lons = $all | ForEach-Object { $_.lon }
    Write-Host ("[stat] lat[{0:f2},{1:f2}] lon[{2:f2},{3:f2}]" -f ($lats | Measure-Object -Minimum).Minimum, ($lats | Measure-Object -Maximum).Maximum, ($lons | Measure-Object -Minimum).Minimum, ($lons | Measure-Object -Maximum).Maximum)
}
