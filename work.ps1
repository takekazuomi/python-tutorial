filter Set-TextUtf8
{
    Param([string] $Path)
    process
    {
        [Text.Encoding]::UTF8.GetBytes($_) | Set-Content -Path $Path -Encoding Byte 
    }
}


Get-ChildItem | ConvertTo-Json | Set-TextUtf8 Out-String -Path t.txt 
