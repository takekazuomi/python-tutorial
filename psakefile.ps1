Task default -Depends CreateNewEnv

# overwrite from command line
# psake -properties @{"name" = "myenv2"}

Properties {
    $name = "myenv"
}


Task CreateNewEnv {
    Exec {conda create -n $name scikit-learn=0.20 python=3.7 pandas jupyter keyring matplotlib joblib pyodbc sqlalchemy}
    Exec {activate.ps1 $name}
    conda config --env --append channels conda-forge
    conda install jupyter_contrib_nbextensions
    pip install lightgbm
}