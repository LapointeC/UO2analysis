from local_pymatgen.io.ase import AseAtomsAdaptor
from local_pymatgen.core.surface import SlabGenerator
from ase.build import bulk
import time

ini_t = time.process_time()
bulk_atoms = bulk('Cu', 'fcc', a=3.6)*[3,3,3]
#print(len(bulk_atoms))
bulk_pymatgen = AseAtomsAdaptor.get_structure(bulk_atoms)
gen = SlabGenerator(bulk_pymatgen, [1,1,0], 10.0, 10.0, lll_reduce=False,
                        center_slab=True, in_unit_planes=True, max_normal_search=10)
gen.get_slabs(ftol=1e-3, tol=1e-3)
end_t = time.process_time()
#print(end_t-ini_t)
