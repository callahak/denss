import sys, os, time
import numpy as np
from scipy import spatial
import saxstats.saxstats as saxs
import matplotlib.pyplot as plt

try: 
    import numba as nb
    numba = True
    #suppress some unnecessary deprecation warnings
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
except:
    numba = False

print("Using Numba: %s"%numba)

def pdb2sas(pdb, q=np.linspace(0,0.5,501),shannon=True):
    """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
    
    This function is simple, only for theoretical tests, and assumes a vacuum
    (i.e. no solvent considerations) and that all atoms are carbon.
    pdb - a saxstats PDB file object.
    q - q values to use for calculations. Default 0.001 < q < 0.500, with 500 data points.
    """
    rij = spatial.distance.squareform(spatial.distance.pdist(pdb.coords[:,:3]))
    D = rij.max()+(2*1.7)
    np.clip(rij, 1e-10, D, out=rij)
    natoms = pdb.natoms
    if shannon:
        wsh = np.pi / D
        nsh = np.ceil(q.max() / wsh).astype(int) + 5
        qsh = (np.arange(nsh)+1)*wsh
        ff = np.zeros((natoms,nsh))
        for i in range(natoms):
            ff[i,:] = saxs.formfactor(pdb.atomtype[i],q=qsh)
        if numba:
            Ish = _pdb2sas_nb(rij, qsh, ff)
        else:
            Ish = _pdb2sas(rij, qsh, ff)
        I = Ish2Iq(Ish=Ish,D=D,q=q)
    else:
        ff = np.zeros((natoms,len(q)))
        for i in range(natoms):
            ff[i,:] = saxs.formfactor(pdb.atomtype[i],q=q)
        if numba:
            I = _pdb2sas_nb(rij, q, ff)
        else:
            I = _pdb2sas(rij, q, ff)
    Iq = np.zeros((len(q),3))
    Iq[:,0] = q
    Iq[:,1] = I
    Iq[:,2] = np.abs(np.random.normal(loc=0.0,scale=0.003*I[0],size=I.shape)) + 0.003*I[0]
    return Iq

def Ish2Iq(Ish,D,q=(np.arange(500)+1.)/1000):
    """Calculate I(q) from intensities at Shannon points."""
    n = len(Ish)
    N = np.arange(n)+1
    denominator = (N[:,None]*np.pi)**2-(q*D)**2
    I = 2*np.einsum('k,ki->i',Ish,(N[:,None]*np.pi)**2 / denominator * np.sinc(q*D/np.pi) * (-1)**(N[:,None]+1))
    return I

def _pdb2sas(rij, q, ff):
    """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
    This function is slower than the similar function implemented with numba.
    rij - distance matrix, ie. output from scipy.spatial.distance.pdist(pdb.coords) (after squareform)
    q - q values to use for calculations. Default 0.001 < q < 0.500, with 500 data points.
    ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
    """
    s = np.sinc(q * rij[...,None]/np.pi)
    I = np.einsum('iq,jq,ijq->q',ff,ff,s)
    return I

if numba:
    @nb.njit(fastmath=True,parallel=True,error_model="numpy",cache=True)
    def _pdb2sas_nb(rij, q, ff):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
        This function is intended to be used with the numba njit decorator for speed.
        rij - distance matrix, ie. output from scipy.spatial.distance.pdist(pdb.coords) (after squareform)
        q - q values to use for calculations. Default 0.001 < q < 0.500, with 500 data points.
        ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
        """
        nr = rij.shape[0]
        nq = q.shape[0]
        I = np.empty(nq)
        ff_T = np.ascontiguousarray(ff.T)
        for qi in nb.prange(nq):
            acc=0
            for ri in range(nr):
                for rj in range(nr):
                    #acc += ff[ri,qi]*ff[rj,qi]*np.sinc(q[qi]*rij[ri,rj]/np.pi)
                    if q[qi]*rij[ri,rj] != 0:
                        acc += ff_T[qi,ri]*ff_T[qi,rj]*np.sin(q[qi]*rij[ri,rj])/(q[qi]*rij[ri,rj])
                    else:
                        acc += ff_T[qi,ri]*ff_T[qi,rj]
            I[qi]=acc
        return I

pdb = saxs.PDB(sys.argv[1])
basename, ext = os.path.splitext(sys.argv[1])

#calculate profile at these q values:
q = np.linspace(0,2,1001)
shannon = True

start = time.time()
Iq = pdb2sas(pdb,q=q,shannon=shannon)
end = time.time()
print("Total calculation time: %s" % (end-start))
np.savetxt(basename+'.pdb2sas.dat',Iq,delimiter=' ',fmt='%.8e')







