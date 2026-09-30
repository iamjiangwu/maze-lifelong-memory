$ErrorActionPreference = 'Stop'
$deckRoot = 'D:\combineddata\github_clean_repo\tmp\ed02_round13_g_labels_2026-08-09\deck'
$jobs = @(
    @{ Input = Join-Path $deckRoot 'sup-figs-v2_ed02-slide3_round13_review.pptx'; Output = Join-Path $deckRoot 'powerpoint-render-round13' },
    @{ Input = Join-Path $deckRoot 'ed02-slide3_round13_review.pptx'; Output = Join-Path $deckRoot 'one-slide-powerpoint-render-round13' },
    @{
        Input = 'D:\combineddata\github_clean_repo\docs\figure_version_records\extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor\review_evidence\sup-figs-v2_ed02-slide3_round12_review.pptx'
        Output = Join-Path $deckRoot 'baseline-powerpoint-render-round12'
    }
)
foreach ($job in $jobs) {
    $inputResolved = [System.IO.Path]::GetFullPath($job.Input)
    $outputResolved = [System.IO.Path]::GetFullPath($job.Output)
    $allowedInputRoots = @(
        $deckRoot,
        'D:\combineddata\github_clean_repo\docs\figure_version_records\extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor\review_evidence'
    )
    if (-not ($allowedInputRoots | Where-Object { $inputResolved.StartsWith($_, [System.StringComparison]::OrdinalIgnoreCase) })) {
        throw "Input escaped the intended roots: $inputResolved"
    }
    if (-not $outputResolved.StartsWith($deckRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Output escaped the intended deck root: $outputResolved"
    }
    $powerPoint = New-Object -ComObject PowerPoint.Application
    try {
        New-Item -ItemType Directory -Force -Path $outputResolved | Out-Null
        $presentation = $powerPoint.Presentations.Open($inputResolved, $true, $true, $false)
        try { $presentation.Export($outputResolved, 'PNG', 1440, 1440) }
        finally { $presentation.Close() }
    }
    finally {
        $powerPoint.Quit()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
    }
}
