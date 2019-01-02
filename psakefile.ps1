Task default -Depends CreateNewEnv

# overwrite from command line
# psake -properties @{"name" = "myenv2"}

Properties {
    $name = "myenv"
    $packages = @("scikit-learn", "python=3.7", "pandas", "jupyter", "keyring", "matplotlib", "joblib", "pyodbc", "sqlalchemy")
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

