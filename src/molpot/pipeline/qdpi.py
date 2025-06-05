from pathlib import Path

import numpy as np
import requests
import torch

import molpot as mpot
from molpot import alias
from collections import defaultdict
import h5py

logger = mpot.get_logger("molpot.dataset")
config = mpot.get_config()

class QDpi:

    urls = {
        "charged": {
            "re_charged": "charged/re_charged.hdf5",
            "remd_charged": "charged/remd_charged.hdf5",
            "spice_charged": "charged/spice_charged.hdf5",
        },
        "neutral": {
            "ani": "neutral/ani.hdf5",
            "comp6": "neutral/comp6.hdf5",
            "freesolvmd": "neutral/freesolvmd.hdf5",
            "geom": "neutral/geom.hdf5",
            "re": "neutral/re.hdf5",
            "remd": "neutral/remd.hdf5",
            "spice": "neutral/spice.hdf5",
        }
    }

    _symbol_to_Z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
        "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
        "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
        "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54
    }

    def __init__(
        self,
        save_dir: Path | None = None,
        device: str = "cpu",
        subset: str | list[str] = "all",
    ):
        from .dataset import MapStyleDataset

        self.name = "QDpi"
        self.save_dir = save_dir
        self.device = device
        self.subset = subset
        self._frames: list[mpot.frames] = []
        self._prepared = False

        if not issubclass(self.__class__, MapStyleDataset):
            self.__class__ = type("QDpi", (self.__class__, MapStyleDataset), {})
            MapStyleDataset.__init__(self, self.name, save_dir=self.save_dir, device=self.device)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def get_subset_data(self):
        if self.subset == "all":
            return self.urls

        ds = defaultdict(dict)
        missing = []

        def add_entry(key: str):
            for category in self.urls:
                if key in self.urls[category]:
                    ds[category][key] = self.urls[category][key]
                    return True
            return False

        if isinstance(self.subset, str):
            if not add_entry(self.subset):
                missing.append(self.subset)
        elif isinstance(self.subset, list):
            for key in self.subset:
                if not add_entry(key):
                    missing.append(key)

        if missing:
            print(f"[Warning] The following entries were not found: {missing}")

        return dict(ds)

    def prepare(self):
        logger.info("prepaering QDpi dataset...")
        if self._prepared:
            return self._frames
        self._prepared = True
        ds = self.get_subset_data()
        for category, subds in ds.items():
            for key, url in subds.items():
                self._download(url, key)
        return self._frames

    def _download(self, url: str, key: str):
        gitlab_root = "https://gitlab.com/RutgersLBSR/QDpiDataset/-/raw/main/data/"
        qdpi_hdf5 = Path(self.save_dir) / f"{key}.hdf5"

        need_download = True
        if qdpi_hdf5.exists():
            try:
                with h5py.File(qdpi_hdf5, "r") as f:
                    for mol_name, mol in f.items():
                        for req_key in ("type.raw", "type_map.raw"):
                            if req_key not in mol:
                                raise RuntimeError
                need_download = False
                logger.info(f"{qdpi_hdf5.name} already exists and is valid, skipping download")
            except Exception:
                try:
                    qdpi_hdf5.unlink()
                except OSError:
                    pass

        if need_download:
            with requests.get(gitlab_root + url, stream=True) as response:
                response.raise_for_status()
                qdpi_hdf5.parent.mkdir(parents=True, exist_ok=True)
                with open(qdpi_hdf5, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logger.info(f"Downloaded {qdpi_hdf5.name}")

        with h5py.File(qdpi_hdf5, "r") as f:
            for mol_name, mol in f.items():
                for req_key in ("type.raw", "type_map.raw"):
                    if req_key not in mol:
                        raise RuntimeError
                coord = torch.tensor(np.array(mol["set.000"]["coord.npy"])).reshape(-1, 3)
                energy = torch.tensor(np.array(mol["set.000"]["energy.npy"]))
                force = torch.tensor(np.array(mol["set.000"]["force.npy"])).reshape(-1, 3)
                
                # Check if net_charge exists, use default value if not
                if "net_charge.npy" in mol["set.000"]:
                    net_charge = torch.tensor(np.array(mol["set.000"]["net_charge.npy"]))
                else:
                    # Default to neutral charge if not specified
                    net_charge = torch.tensor(0.0)
                    logger.info(f"No net_charge found for {mol_name}, defaulting to 0.0")

                type_raw_np = np.array(mol["type.raw"], dtype=int)
                type_map_raw = mol["type_map.raw"][()]
                decoded = [b.decode("utf-8").rstrip("\x00").strip() for b in type_map_raw]
                type_map_nums = np.array([self._symbol_to_Z[sym] for sym in decoded], dtype=int)
                atom_numbers = type_map_nums[type_raw_np]
                type_name = torch.tensor(atom_numbers, dtype=torch.long)

                frame = mpot.Frame()
                frame[alias.R] = coord
                frame[alias.F] = force
                frame[alias.E] = energy
                frame[alias.Q] = net_charge
                frame[alias.Z] = type_name

                self._frames.append(frame)

        return self._frames
