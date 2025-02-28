# How To

Clone [my branch](https://github.com/devrandom/open-webui/tree/devrandom-static-plugins) of open-webui and install this repo as follows:

```sh
git clone -b devrandom-static-plugins git@github.com:devrandom/open-webui.git
git clone git@github.com:devrandom/open-webui-plugins.git plugins
(cd plugins && uv pip install .)
```

Then start open-webui with the follow environment variable set:

```sh
export OPEN_WEBUI_PLUGINS=open_webui_plugins
```

The plugins in this repository will be automatically loaded on startup.

# Plugins
## Mem

Automatically add memories when detected in user messages.
