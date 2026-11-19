"""Version checking and package metadata utilities for Gannet."""

import importlib.util
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class GannetVersionInfo:
    """Container holding Gannet and module version identifiers.

    Attributes
    ----------
    gannet: str
        Semantic version string for Gannet.
    load: str
        Module version date code for GannetLoad.
    fit: str
        Module version date code for GannetFit.
    fit_phantom: str
        Module version date code for GannetFitPhantom.
    coreg: str
        Module version date code for GannetCoRegister.
    segment: str
        Module version date code for GannetSegment.
    quantify: str
        Module version date code for GannetQuantify.
    coregstandalone: str
        Module version date code for CoRegStandAlone.
    """

    gannet: str = "3.5.4"
    load: str = "260814"
    fit: str = "260618"
    fit_phantom: str = "240207"
    coreg: str = "250914"
    segment: str = "260118"
    quantify: str = "251223"
    coregstandalone: str = "260109"


def gannet_version(print_output: bool = True) -> GannetVersionInfo:
    """Return the version numbers of Gannet and its modules.

    Parameters
    ------------
    print_output: bool
        Whether to print formatted version information to stdout.
        Defaults to True.

    Returns
    --------
        GannetVersionInfo: dataclass object
            Version metadata for Gannet and all core pipeline modules.
    """
    info = GannetVersionInfo()
    if print_output:
        print("\n--------------------------------------")
        print(f"Your version of Gannet is {info.gannet}")
        print("--------------------------------------")
        print(f"GannetLoad:           {info.load}")
        print(f"GannetFit:            {info.fit}")
        print(f"GannetCoRegister:     {info.coreg}")
        print(f"GannetSegment:        {info.segment}")
        print(f"GannetQuantify:       {info.quantify}\n")
    return info


def toolbox_check() -> list[str]:
    """Verify that all required Python scientific dependencies are installed.

    Returns
    --------
        list[str]: list of str
            List of verified installed package names.

    Raises
    ------
    ImportError
        If any required dependency cannot be imported.
    """
    required = ["numpy", "scipy", "matplotlib", "nibabel", "pydicom", "pandas"]
    missing: list[str] = []
    verified: list[str] = []
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
        else:
            verified.append(pkg)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            "The following required Python packages are missing and need to "
            f"be installed to run Gannet: {joined}"
        )
    return verified


def version_check(silent: bool = False, current_version: str = "") -> dict[str, object]:
    """Check whether a newer version of Gannet is available on GitHub.

    Parameters
    ------------
    silent: bool
        If True, suppress console warnings when offline or outdated.
        Defaults to False.
    current_version: str
        Current version string to compare against upstream. Defaults to "".

    Returns
    --------
        dict[str, object]: dict object
            Dictionary containing 'current_version' and 'new_version_available'.
    """
    if not current_version:
        current_version = GannetVersionInfo().gannet
    new_version_available = False
    url = "https://raw.githubusercontent.com/markmikkelsen/Gannet/main/GannetLoad.m"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            text = response.read().decode("utf-8", errors="ignore")
        match = re.search(r"MRS_struct\.info\.version\.Gannet\s*=\s*'(.*?)'", text)
        if match:
            latest_version = match.group(1)
            latest_num = int("".join(re.findall(r"\d", latest_version)) or "0")
            curr_num = int("".join(re.findall(r"\d", current_version)) or "0")
            if latest_num > curr_num:
                new_version_available = True
                if not silent:
                    print(
                        "\n*********************************************"
                        "**************************************************"
                    )
                    print(
                        f"A newer version of Gannet ({latest_version}) is "
                        f"available. You are currently running version "
                        f"{current_version}."
                    )
                    print(
                        "You can download the latest version from "
                        "https://github.com/markmikkelsen/Gannet."
                    )
                    print(
                        "*************************************************"
                        "**********************************************\n"
                    )
    except (urllib.error.URLError, TimeoutError, OSError):
        if not silent:
            pass
    result: dict[str, object] = {
        "current_version": current_version,
        "new_version_available": new_version_available,
    }
    return result


def update_gannet() -> str:
    """Provide guidance on updating the Gannet Python package via uv.

    Returns
    --------
        str: str object
            Instruction message for updating the package.
    """
    msg = (
        "To update the Python Gannet package, pull the latest repository "
        "changes and run 'uv sync --upgrade'."
    )
    print(msg)
    return msg
