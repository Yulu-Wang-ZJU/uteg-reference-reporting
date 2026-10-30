$ErrorActionPreference = 'Stop'
$base = Split-Path $PSScriptRoot -Parent
$src = Join-Path $base 'figures/Figure1_workflow_ma.pptx'
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pp.WindowState = 2
    $pres = $pp.Presentations.Open($src, $false, $false, $false)
    $pres.Slides.Item(1).Export((Join-Path $base 'figures/Figure1_workflow_ma.png'), 'PNG', 1600, 1280)
    $pres.SaveAs((Join-Path $base 'figures/Figure1_workflow_ma.pdf'), 32)
    Write-Output 'Aligned Figure 1 exported from retained native source.'
}
finally {
    if ($null -ne $pres) { $pres.Close() }
    $pp.Quit()
}
