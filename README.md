# Python とか、scikit-learnの学習用

参考リンク

1. [scikit-learn](https://scikit-learn.org/)
2. [python](https://scikit-learn.org/)
3. [conda](https://conda.io/docs/index.html)

## 環境

1. mini conda を入れる。 [mini conda](https://conda.io/miniconda.html)
2. psenvに、mini conda の 設定を追加。

## conda 設定

参考：
[conda Managing environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)

1. 環境の作成

名前をつけて仮想環境を作成する。同時に、インストールするパッケージを指定出来る。

```posh
conda create -n myenv scikit-learn=0.20 python=3.7 pandas jupyter keyring matplotlib joblib sqlalchemy
```

jupyter の nbextensionsが便利なので、インストールする。これは、別チャンネルなので、環境作成後、環境のchannnel に追加してから、インストールする。
conda config に--name が無いので、環境を切り替えてから（activate してから）、conda config --env を実行する。
コアパッケージを固定して、無いものだけ他から持ってくるためには、環境のchannnel に低優先度でconda-forgeを追加しておく。これが重要。

```posh
activate myenv
(myenv) $ conda config --env --append channels conda-forge
```

この設定は環境の保存先にファイルが作られて保存される。これだと、レポジトリには入らないので注意。

```posh
(myenv) $ conda config  --show-sources
==> C:\Users\takekazu\.condarc <==
channels:
  - defaults

==> C:\opt\Miniconda3\envs\myenv\.condarc <==
channels:
  - defaults
  - conda-forge
```

NG例：コマンドラインでチャンネルを指定した時

```posh
conda install -n myenv -c conda-forge jupyter_contrib_nbextensions
(myenv4) $ conda install -n myenv -c conda-forge jupyter_contrib_nbextensions
Solving environment: done

## Package Plan ##

  environment location: C:\opt\Miniconda3\envs\myenv

  added / updated specs:
    - jupyter_contrib_nbextensions


The following packages will be UPDATED:

    ca-certificates: 2018.03.07-0         --> 2018.11.29-ha4d7672_0 conda-forge
    certifi:         2018.11.29-py37_0    --> 2018.11.29-py37_1000  conda-forge

The following packages will be DOWNGRADED:

    openssl:         1.1.1a-he774522_0    --> 1.0.2p-hfa6e2cd_1001  conda-forge
    qt:              5.9.7-vc14h73c81de_0 --> 5.9.6-vc14h1e9a669_2

Proceed ([y]/n)?
```

conda-forgeをコマンドラインで指定すると、別のビルドのpython 3.7.1が入り、いくつかのパッケージが更新される。コマンドラインで指定したチャンネルが優先されるからのようだ。

環境のチャンネルに追加しておいて、コマンドラインで指定しなかったときには、こうなる

```posh
(myenv) $ conda install jupyter_contrib_nbextensions
Solving environment: done

## Package Plan ##

  environment location: C:\opt\Miniconda3\envs\myenv4

  added / updated specs:
    - jupyter_contrib_nbextensions


The following NEW packages will be INSTALLED:

    jupyter_contrib_core:              0.3.3-py_2           conda-forge
    jupyter_contrib_nbextensions:      0.5.0-py37_1000      conda-forge
    jupyter_highlight_selected_word:   0.2.0-py37_1000      conda-forge
    jupyter_latex_envs:                1.4.4-py37_1000      conda-forge
    jupyter_nbextensions_configurator: 0.4.0-py37_1000      conda-forge
    libiconv:                          1.15-h1df5818_7
    libxml2:                           2.9.8-hadb2253_1
    libxslt:                           1.1.32-hf6f1972_0
    lxml:                              4.2.5-py37hef2cd61_0
    pyyaml:                            3.13-py37hfa6e2cd_0
    yaml:                              0.1.7-hc54c509_2

Proceed ([y]/n)?
```

コマンドラインでチャンネルを指定した場合と、環境に低優先度でチャンネルを追加した場合で動作が違う。環境のチャンネルに低優先度で入れておいた方が期待通りの動きになる。

コマンドラインで-cを指定した場合、そのチャンネルを優先してパッケージを持ってくる。指定しない場合は、環境のチャンネルリストを使って、パッケージを解決する。
参考：coda 4.1以降 <https://conda.io/docs/user-guide/tasks/manage-channels.html#after-conda-4-1-0> は、コアパッケージが上書きされなくなるように、チャンネルリストでのプライオリティ付が使われるようになった。
コアパッケージをdefaults(anaconda)のものにしたかったら、```conda config --env --append channels``` で低優先度で必要なチャンネルを追加する。コマンドラインで-c で指定したチャンネルを低優先度にする方法は無く、必ず最優先になるので、コマンドラインで指定するとコアパッケージが更新されてしまう。TODO: ドキュメントを探す。

## nbextensionsのパッチ

nbextensions がタブに出てこないBUGがあるようなので、気になるようならソースにパッチする。
<https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator/pull/85/files>
master にマージされてるけど、conda-forge (0.5.0)が更新されていない。
<https://anaconda.org/conda-forge/jupyter_nbextensions_configurator>

参照: <https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator/issues/72#issuecomment-437251856>

手元の環境だと下記のパスのファイルを直せば表示される。

```posh
c:/opt/Miniconda3/envs/myenv/Lib/site-packages/jupyter_nbextensions_configurator/static/nbextensions_configurator/tree_tab/main.js
```

試しに、全部更新してみる。特になにも更新されない。入れたばっかりなので、こうでないといけない。

```posh
(myenv) $ conda update --all
Solving environment: done

# All requested packages already installed.
```

## scikit-learn 0.20.2

最初に入れるときに、20.1を指定しておいて、20.2への更新を確認する。

```posh
psake CreateNewEnv -properties @{"name" = "myenv3"}
(myenv3) $ conda list | sls scikit

scikit-learn              0.20.1           py37h343c172_0
```

scikit-learn 0.20.2 になった、リリースノート <https://scikit-learn.org/dev/whats_new.html#version-0-20-2>

```posh
(myenv3) $ conda update --all
Solving environment: done

## Package Plan ##

  environment location: C:\opt\Miniconda3\envs\myenv3


The following packages will be UPDATED:

    scikit-learn: 0.20.1-py37h343c172_0 --> 0.20.2-py37h343c172_0

Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

numpyがロードできるかどうかを確認。ついでにコンフィグレーションも。この環境ではmklが使われる。

```posh
(myenv3) $ python -c 'import numpy as np;np.show_config()'
mkl_info:
    libraries = ['mkl_rt']
    library_dirs = ['C:/opt/Miniconda3/envs/myenv\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/opt/Miniconda3/envs/myenv\\Library\\include']
blas_mkl_info:
    libraries = ['mkl_rt']
    library_dirs = ['C:/opt/Miniconda3/envs/myenv\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/opt/Miniconda3/envs/myenv\\Library\\include']
blas_opt_info:
    libraries = ['mkl_rt']
    library_dirs = ['C:/opt/Miniconda3/envs/myenv\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/opt/Miniconda3/envs/myenv\\Library\\include']
lapack_mkl_info:
    libraries = ['mkl_rt']
    library_dirs = ['C:/opt/Miniconda3/envs/myenv\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/opt/Miniconda3/envs/myenv\\Library\\include']
lapack_opt_info:
    libraries = ['mkl_rt']
    library_dirs = ['C:/opt/Miniconda3/envs/myenv\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/opt/Miniconda3/envs/myenv\\Library\\include']
```

コアパッケージが混ざると、更新したこときにおかしくなるので注意する。

see. <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html#conda>


##　lightgbmを入れる

```posh
pip install lightgbm

```


