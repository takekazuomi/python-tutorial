Task default -Depends CreateNewEnv

# overwrite from command line
# psake -properties @{"name" = "myenv2"}

Properties {
    $name = "myenv"
    $packages = @("scikit-learn", "python=3.7", "pandas", "jupyter", "keyring", "matplotlib", "joblib", "pyodbc", "sqlalchemy")
}

filter Set-TextUtf8
{
    Param([string] $Path)
    process
    {
        [Text.Encoding]::UTF8.GetBytes($_) | Set-Content -Path $Path -Encoding Byte 
    }
    end {

    }
}

Task CreateNewEnv {
    Exec {conda create -y -n $name $packages }
    activate.ps1 $name
    conda config --env --append channels conda-forge
    conda install -y jupyter_contrib_nbextensions
    pip install lightgbm
    conda env export > "$name.yml"
}

Task CleanEnv {
    conda info -e --json | convertfrom-json | Select-Object -ExpandProperty Envs | split-path -Leaf | ?{ $_ -like "$name*"} | %{conda remove -n $_ --all}
}

Task MakeModel -description "Generate model" {
    $d = Get-Date -f 'yyyymmddhhMM'
    $j = python.exe ./predict_kaijo.py -m -j -k 2 | ConvertFrom-Json
    ConvertTo-Json -InputObject @($j) | Set-TextUtf8 -Path "model/$d-training.json"
}

Task MakeModelForKaijo -description "Generate model for each kaijo" {
    $d = Get-Date -f 'yyyymmddhhMM'
    1..24 | %{
    	python.exe ./predict_kaijo.py -m -k $_ -j | convertfrom-json
    } | ConvertTo-Json | Set-TextUtf8 -Path "model/$d-training.json"
}

Task ValidationModelForKaijo -description "Validation Kaijo Model" {
    $d = Get-Date -f 'yyyymmddhhMM'
    #$m = ".\model\201812252013-training.json" #LightGBM チューニング無し
    $m = ".\model\201846270612-training.json" #LightGBM チューニング、n_estimators=1000, max_bin=1000
    $t = "predictv2"
    python.exe .\predict_kaijo.py -v $m -t $t -j -k model -w
}


