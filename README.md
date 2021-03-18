# auction
Ad auction in Python


### Install scipy on OSX

```
% brew install openblas

~/promotedai/auction % SYSTEM_VERSION_COMPAT=1 \
LDFLAGS="-L/usr/local/opt/openblas/lib" \
CPPFLAGS="-I/usr/local/opt/openblas/include" \
OPENBLAS="$(brew --prefix openblas)" \
pipenv install scipy
```