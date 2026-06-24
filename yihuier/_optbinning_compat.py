"""optbinning 运行时兼容垫片。

optbinning（截至 0.20.0，且项目已停止维护）在多处通过
``from sklearn.utils import check_array`` 绑定并调用 ``check_array`` 时传入
``force_all_finite`` 关键字。该参数在 scikit-learn 1.6 中被重命名为
``ensure_all_finite``，因此 sklearn>=1.6 下 optbinning 会在 fit 时抛
``TypeError``。

本模块把 ``force_all_finite`` 翻译为 ``ensure_all_finite``，使 optbinning
在新版 sklearn 下仍可用。

注意：optbinning 在导入时即绑定了 ``check_array`` 引用，因此本模块需要：
  1. 替换 ``sklearn.utils`` / ``sklearn.utils.validation`` 中的 ``check_array``
     （覆盖 shim 之后才 import optbinning 的情况）；
  2. 若 optbinning 已经被导入，遍历其子模块并替换已绑定的 ``check_array``
     （覆盖 optbinning 先于 shim 导入的情况，如测试/其它代码先 import 了它）。

幂等：重复导入只打补丁一次（``_yihuier_force_finite_patched`` 标志）。
"""

import importlib
import pkgutil

import sklearn.utils as _su
import sklearn.utils.validation as _sv

_FLAG = "_yihuier_force_finite_patched"

if not getattr(_sv, _FLAG, False):
    _orig_check_array = _sv.check_array

    def _translate_finite_kwarg(kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return kwargs

    def check_array(*args, **kwargs):
        return _orig_check_array(*args, **_translate_finite_kwarg(kwargs))

    # 1) 替换 sklearn 中的引用（覆盖 shim 之后才导入 optbinning 的情况）
    _sv.check_array = _su.check_array = check_array

    # 2) 若 optbinning 已被导入，替换其子模块已绑定的 check_array
    try:
        import optbinning as _ob

        for _mod_info in pkgutil.walk_packages(_ob.__path__, _ob.__name__ + "."):
            try:
                _mod = importlib.import_module(_mod_info.name)
            except Exception:
                continue
            # 仅替换指向 sklearn 的 check_array，避免误伤同名无关属性
            _attr = getattr(_mod, "check_array", None)
            if _attr is _orig_check_array or (
                callable(_attr) and getattr(_attr, "__module__", "").startswith("sklearn")
            ):
                _mod.check_array = check_array
    except ImportError:
        pass

    setattr(_sv, _FLAG, True)
