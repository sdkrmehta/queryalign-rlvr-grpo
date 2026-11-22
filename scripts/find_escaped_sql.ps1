Get-ChildItem -Path "c:\queryalign-rlvr-grpo\notebooks" -Filter "*.ipynb" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    $count = ([regex]::Matches($content, '<\\/sql>')).Count
    Write-Host "$($_.Name): $count occurrences of '<\/sql>'"
}
