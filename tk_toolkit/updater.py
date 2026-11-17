"""Updates from GitHub releases: check the latest release and install its zip in place."""

import json
import os
import re
import ssl
import tempfile
import urllib.request

import bpy
from bpy.props import BoolProperty, StringProperty

from . import common as C

DEFAULT_REPO = "sightranvolodin-oss/tk-toolkit"
_state = {"latest": None, "url": None, "notes": "", "error": "", "checked": False}


def addon_package():
    return __package__


def current_version():
    """Version from blender_manifest.toml (extension) or bl_info (legacy add-on)."""
    path = os.path.join(os.path.dirname(__file__), "blender_manifest.toml")
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
            if m:
                return m.group(1)
    except OSError:
        pass
    from . import bl_info
    return ".".join(str(x) for x in bl_info["version"])


def parse_version(s):
    nums = re.findall(r"\d+", s or "")
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def prefs(context=None):
    ctx = context or bpy.context
    a = ctx.preferences.addons.get(addon_package())
    return a.preferences if a else None


def online_allowed():
    return getattr(bpy.app, "online_access", True)


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "tk-toolkit-updater"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def check(repo):
    """Fill _state from the repo's latest release. Returns True if newer exists."""
    _state.update(latest=None, url=None, notes="", error="", checked=True)
    try:
        data = json.loads(_get(f"https://api.github.com/repos/{repo}/releases/latest"))
    except Exception as ex:  # network, 404 when no releases yet, rate limit…
        msg = str(ex)
        if "404" in msg:
            msg = "в репозитории пока нет релизов"
        _state["error"] = msg
        return False
    tag = data.get("tag_name", "")
    asset = next((a for a in data.get("assets", []) if a.get("name", "").endswith(".zip")), None)
    _state["latest"] = tag.lstrip("v")
    _state["url"] = asset.get("browser_download_url") if asset else None
    _state["notes"] = (data.get("body") or "").strip()
    if asset is None:
        _state["error"] = "в последнем релизе нет zip-файла"
    return parse_version(_state["latest"]) > parse_version(current_version())


class TK_OT_update_check(bpy.types.Operator):
    bl_idname = "tk.update_check"
    bl_label = "Проверить обновления"
    bl_description = "Посмотреть последний релиз аддона на GitHub"

    def execute(self, context):
        if not online_allowed():
            self.report({"WARNING"}, "Доступ в интернет выключен: Preferences → System → Network")
            return {"CANCELLED"}
        p = prefs(context)
        repo = p.repo if p else DEFAULT_REPO
        newer = check(repo)
        if _state["error"] and not _state["latest"]:
            self.report({"WARNING"}, "Не удалось проверить: " + _state["error"])
        elif newer:
            self.report({"INFO"}, f"Доступна версия {_state['latest']} (у вас {current_version()})")
        else:
            self.report({"INFO"}, f"У вас последняя версия ({current_version()})")
        return {"FINISHED"}


class TK_OT_update_install(bpy.types.Operator):
    bl_idname = "tk.update_install"
    bl_label = "Установить обновление"
    bl_description = "Скачать zip последнего релиза и установить поверх текущей версии"

    @classmethod
    def poll(cls, context):
        return bool(_state.get("url"))

    def execute(self, context):
        if not online_allowed():
            self.report({"WARNING"}, "Доступ в интернет выключен: Preferences → System → Network")
            return {"CANCELLED"}
        tmp = os.path.join(tempfile.mkdtemp(prefix="tk_update_"),
                           f"tk_toolkit-{_state['latest']}.zip")
        try:
            with open(tmp, "wb") as f:
                f.write(_get(_state["url"], timeout=60))
        except Exception as ex:
            self.report({"ERROR"}, f"Не удалось скачать: {ex}")
            return {"CANCELLED"}
        pkg = addon_package()
        if pkg.startswith("bl_ext."):
            repo_module = pkg.split(".")[1]
            try:
                bpy.ops.extensions.package_install_files(
                    filepath=tmp, repo=repo_module, enable_on_install=True, overwrite=True)
            except TypeError:
                bpy.ops.extensions.package_install_files(
                    filepath=tmp, repo=repo_module, enable_on_install=True)
        else:
            bpy.ops.preferences.addon_install(filepath=tmp, overwrite=True)
        self.report({"INFO"}, f"Установлена версия {_state['latest']}. "
                              "Перезапустите Blender, чтобы всё обновилось начисто")
        return {"FINISHED"}


class TK_OT_open_repo(bpy.types.Operator):
    bl_idname = "tk.open_repo"
    bl_label = "Открыть на GitHub"

    def execute(self, context):
        p = prefs(context)
        bpy.ops.wm.url_open(url=f"https://github.com/{p.repo if p else DEFAULT_REPO}")
        return {"FINISHED"}


def draw_update_ui(layout):
    col = layout.column(align=True)
    col.label(text=f"Версия: {current_version()}", icon="INFO")
    if _state["checked"]:
        if _state["latest"] and parse_version(_state["latest"]) > parse_version(current_version()):
            col.label(text=f"Доступна {_state['latest']}", icon="IMPORT")
            for line in _state["notes"].splitlines()[:6]:
                col.label(text=line[:70])
        elif _state["latest"]:
            col.label(text="Последняя версия", icon="CHECKMARK")
        if _state["error"]:
            col.label(text=_state["error"], icon="ERROR")
    row = layout.row(align=True)
    row.operator("tk.update_check", icon="FILE_REFRESH")
    row.operator("tk.update_install", icon="IMPORT")
    layout.operator("tk.open_repo", icon="URL")


class TK_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    repo: StringProperty(name="Репозиторий GitHub", default=DEFAULT_REPO,
                         description="owner/repo, откуда брать релизы")
    check_on_start: BoolProperty(name="Проверять при запуске", default=False,
                                 description="Раз при старте Blender (если разрешён интернет)")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "repo")
        layout.prop(self, "check_on_start")
        draw_update_ui(layout)


class TK_PT_updates(bpy.types.Panel):
    bl_label = "Обновления"
    bl_idname = "TK_PT_updates"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = C.CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 99

    def draw(self, context):
        draw_update_ui(self.layout)


def _startup_check():
    p = prefs()
    if p and p.check_on_start and online_allowed():
        try:
            check(p.repo)
        except Exception:
            pass
    return None


classes = (TK_OT_update_check, TK_OT_update_install, TK_OT_open_repo,
           TK_AddonPreferences, TK_PT_updates)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    if not bpy.app.background:
        bpy.app.timers.register(_startup_check, first_interval=5.0)


def unregister():
    if bpy.app.timers.is_registered(_startup_check):
        bpy.app.timers.unregister(_startup_check)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
