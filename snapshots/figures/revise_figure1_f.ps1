$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'figures/crosscentre_aligned_20261029'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
  $pres = $pp.Presentations.Open((Join-Path $root 'figures/terminology_aligned_20261029/Figure1_workflow_ma.pptx'), $true, $false, $false)
  $s = $pres.Slides.Item(1)
  $s.Shapes.Item('TITLE').TextFrame.TextRange.Text = 'Agent-based U-TEG translation and selective reporting'
  $s.Shapes.Item('D.heading').TextFrame.TextRange.Text = 'Cross-centre evaluation: Huzhou, Hubei and SAHZU'
  $s.Shapes.Item('D.SAHZU').TextFrame.TextRange.Text = 'Single source'
  $s.Shapes.Item('D.Huzhou').TextFrame.TextRange.Text = 'Two sources'
  $s.Shapes.Item('D.bidirectional').TextFrame.TextRange.Text = '6 directions                  3 directions'
  $s.Shapes.Item('D.metrics.title').TextFrame.TextRange.Text = 'Held-out evaluation'
  $s.Shapes.Item('D.metrics').TextFrame.TextRange.Text = "Agreement and coverage`rMA errors and report time"
  $g = $s.Shapes.Item('D '+[char]0x00B7+' Hubei external validation')
  for ($i=1; $i -le $g.GroupItems.Count; $i++) {
    $x = $g.GroupItems.Item($i)
    if ($x.HasTextFrame -and $x.TextFrame.HasText) {
      if ($x.TextFrame.TextRange.Text -eq 'Hubei') { $x.TextFrame.TextRange.Text = 'Target centre' }
      if ($x.TextFrame.TextRange.Text -eq 'Evaluation complete') { $x.TextFrame.TextRange.Text = 'Separate test cohort' }
    }
  }
  $s.Shapes.Item('AG.draft').Delete()
  $s.Shapes.Item('NEW.endpoint.badge').Delete()
  foreach ($name in @('D.development','D.SAHZU','D.Huzhou','D.bidirectional','D.freeze','D.freeze.label','FLOW.development_freeze','D.separator','D.metrics.title','D.metrics','FLOW.external')) {
    $s.Shapes.Item($name).Top -= 10
  }
  $g.Top -= 10
  $n = $s.Shapes.AddTextbox(1, 35, 928, 1130, 18)
  $n.Name = 'F.main.configurations'
  $n.TextFrame.MarginLeft = 0; $n.TextFrame.MarginRight = 0
  $n.TextFrame.MarginTop = 0; $n.TextFrame.MarginBottom = 0
  $n.TextFrame.TextRange.Text = 'Main comparisons: Huzhou '+[char]0x2192+' SAHZU and Huzhou + Hubei '+[char]0x2192+' SAHZU; same test cohort'
  $n.TextFrame.TextRange.Font.Name = 'Times New Roman'
  $n.TextFrame.TextRange.Font.Size = 14
  $n.TextFrame.TextRange.ParagraphFormat.Alignment = 2
  $pres.SaveAs((Join-Path $out 'Figure1_workflow_ma.pptx'),24)
  $s.Export((Join-Path $out 'Figure1_workflow_ma.png'),'PNG',1600,1280)
  $pres.SaveAs((Join-Path $out 'Figure1_workflow_ma.pdf'),32)
} finally { if ($null -ne $pres) {$pres.Close()}; $pp.Quit() }
