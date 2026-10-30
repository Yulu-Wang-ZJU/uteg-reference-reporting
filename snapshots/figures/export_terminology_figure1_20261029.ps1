$ErrorActionPreference = 'Stop'
$dir = Join-Path (Split-Path $PSScriptRoot -Parent) 'figures/terminology_aligned_20261029'
$src = Join-Path $dir 'Figure1_workflow_ma.pptx'
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pp.WindowState = 2
    $pres = $pp.Presentations.Open($src, $true, $false, $false)
    $pres.Slides.Item(1).Export((Join-Path $dir 'Figure1_workflow_ma.png'), 'PNG', 1600, 1280)
    $pres.SaveAs((Join-Path $dir 'Figure1_workflow_ma.pdf'), 32)
    Write-Output 'Aligned Figure 1 exported; original native source retained.'
}
finally {
    if ($null -ne $pres) { $pres.Close() }
    $pp.Quit()
}
