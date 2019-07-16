""" Dida Markovic, 2018, 2019 
	Some tools to work with Legendre multipoles of the galaxy
	power spectrum and its covariance matrix. """
import numpy as np
from scipy.special import legendre
from scipy.interpolate import interp1d
from scipy.integrate import quad

def covll(kvals,mu,Pkmu,Vs,ng,binaveraged=True):
	""" Calculates the covariance matrix for the galaxy clustering power spectrum multipoles (0,2,4 only).
		Uses equation (91) from https://wwwmpa.mpa-garching.mpg.de/~komatsu/lecturenotes/Shun_Saito_on_RSD.pdf."""

	# The list of multipoles: monopole, quadrupole, hexadecapole
	mlps = [0,2,4]
	
	# Count things
	Nl = len(mlps)
	nk = len(kvals)
	nside = nk*Nl

	# Create the empty matrix
	covmat = np.zeros((nside,nside))

	# Get the k-steps and the mu-steps (if they are not equal, this is going to be a problem!!)
	dk = np.diff(kvals) # Assumes linear k range!!
	dk = np.append(dk,dk[-1]) # Assumes last bin same width as second to last!!

	# Number of k-modes
	Nk = Vs 
	if not np.any(mu<0.): Nk/=2. # Multiply final result by 2 if only have positive mu
	if not binaveraged: 
		Nk *= kvals**2 * dk / (2*np.pi**2)
	else:
		Vk = 4.*np.pi/3 * ( (kvals+dk/2.)**3 - (kvals-dk/2.)**3 )
		pref2 = 2. * (2.*np.pi)**4 / Vk**2
	
	# Loop over all mlps
	kind = np.arange(nk)
	for il,l1 in zip(range(Nl),mlps):
		
		# Calculate the index in the covmat corresponding to the diagonal in k for l1
		xind = kind + il*nk

		for jl,l2 in zip(range(Nl),mlps):

			# Calculate the index in the covmat corresponding to the diagonal in k for l2
			yind = kind + jl*nk

			# Get the full pre-factor
			pref1 = (2.*l1+1.) * (2.*l2+1.) / Nk

			# Set up and calculate the integral for each k-value 
			# Because of the delta function, there are no off-diagonal elemnts in the k-direction,
			# so only 1 k-direction is needed.
			integrand = (Pkmu + 1./ng)**2 * legendre(l1)(mu) * legendre(l2)(mu)
			result = pref1 * np.trapz(integrand,x=mu,axis=1)

			# But our power spectrum is bin-averaged, so need to account for this
			# See equation (16) in https://arxiv.org/pdf/1509.04293.pdf.
			if binaveraged:			
				# Now create an interpolator, so that we can integrate on sub-k-bin-separation scales
				# (not sure how good this is!!)
				interpolator = interp1d(kvals,result*kvals**2,fill_value='extrapolate')

				# Unfortunately probably need to loop over k to do the integrals
				for i,k in enumerate(kvals):
					result[i] = pref2[i]*quad(interpolator,k-dk[i]/2.,k+dk[i]/2.)[0]

			# Now insert into covmat array
			covmat[tuple(xind),tuple(yind)] = result

	return covmat

def covll_duplicate(kvals,mu,Pkmu,Vs,ng,binaveraged=False):
	""" Testing.
		Uses only equation (91) from https://wwwmpa.mpa-garching.mpg.de/~komatsu/lecturenotes/Shun_Saito_on_RSD.pdf. """

	# Assumes only monopole, quadrupole, hexadecapoe
	mlps = [0,2,4]

	# Count things
	nk = len(kvals)
	nl = len(mlps)

	# Get the k-value separation - expecting to be linearly spaced, i.e. equally spaced!
	dk = float_unique(np.diff(kvals),-2) # EXTREMELY LOW TOLERANCE SET!!!
	if len(dk)>1:
		print(dk)
		raise Exception('Can only calculate the covariance matrix for linearly spaced k-values!')
	else:
		dk = dk[0]

	# Calculate the number of k-modes in the volume for each k-mode
	Nk = Vs * kvals**2 * dk / (2.*np.pi**2)

	# Multiply final result by 2 if only have positive mu
	if not np.any(mu<0.): Nk/=2. 

	# Loop over both multipoles
	covmat = np.zeros((nk*nl,nk*nl))
	for l1 in mlps:
		for l2 in mlps:
			prefactor = (2.*l1+1.)*(2.*l2+1.) / Nk
			
			# Loop over k-modes
			for i,k in enumerate(kvals):
				integrand = legendre(l1)(mu) * legendre(l2)(mu) * (Pkmu[i,:]+1./ng)**2
				covmat[l1/2*nk+i,l2/2*nk+i] = prefactor * np.trapz(integrand,x=mu)

	return covmat

def covll_reduced(kvals,mu,Pkmu,ng,mlps=[0,2,4]):
	""" Gets Cov' in equation (91) from https://wwwmpa.mpa-garching.mpg.de/~komatsu/lecturenotes/Shun_Saito_on_RSD.pdf. """

	# Count things
	nk = len(kvals)
	nl = len(mlps)

	# Get the k-value separation - expecting to be linearly spaced, i.e. equally spaced!
	dk = float_unique(np.diff(kvals),-2) # EXTREMELY LOW TOLERANCE SET!!!
	if len(dk)>1:
		print(dk)
		raise Exception('Can only calculate the covariance matrix for linearly spaced k-values!')

	# Multiply final result by 2 if only have positive mu
	pref = 1.
	if not np.any(mu<0.): pref/=2.

	# Loop over both multipoles
	covmat = np.zeros((nk*nl,nk*nl))
	for i1,l1 in enumerate(mlps):
		for i2,l2 in enumerate(mlps):
			prefactor = pref * (2.*l1+1.)*(2.*l2+1.) / 2.
			
			# Loop over k-modes
			for i,k in enumerate(kvals):
				integrand = legendre(l1)(mu) * legendre(l2)(mu) * (Pkmu[i,:]+1./ng)**2
				covmat[i1*nk+i,i2*nk+i] = prefactor * np.trapz(integrand,x=mu)

	return covmat

def pkto2d(k,pk1d,mlps,Nmu=100): # VERIFIED + VALIDATED
	""" Converts from multipoles to a 2d power spectrum. Sets mu to a linear range from 0 to 1.
		Uses equation (21) from https://wwwmpa.mpa-garching.mpg.de/~komatsu/lecturenotes/Shun_Saito_on_RSD.pdf."""

	index = get_k_once(k,index=True)
	nk = len(k[index])

	mu = 2.*np.arange(Nmu)/Nmu - 1. + 1./Nmu
	pk2d = np.zeros((nk,Nmu))

	for i,l in enumerate(mlps):
		pk2d += np.outer(pk1d[index+i*nk],legendre(i*2)(mu))

	return k[index],mu,pk2d

def pkto1d(k,mu,pk2d,mlps=[0,2,4]): # VERIFIED + VALIDATED
	""" Converts 2d power spectrum to multipoles. 
		Uses equation (22) from https://wwwmpa.mpa-garching.mpg.de/~komatsu/lecturenotes/Shun_Saito_on_RSD.pdf."""
	
	# Make one long vector of the multipoles
	nk = len(k)
	nside = nk*len(mlps)
	pk1d = np.zeros(nside)
	newk = np.zeros(nside)
	for i,l in enumerate(mlps):
		pref = 0.5*(2.*l+1.)
		integrand = pk2d * legendre(l)(mu)
		pk1d[i*nk:(i+1)*nk] = pref * np.trapz(integrand,x=mu,axis=1)
		newk[i*nk:(i+1)*nk] = k

	return newk, pk1d

# Extract only one of the repeated k-arrays (return the index - same assumption as above)
# This function is copied from plotlib.py of my EFToBOSS plotting module.
def get_k_once(kvals, index=False): # VERIFIED + VALIDATED
    ends = [len(kvals)]
    ends[:0] = [i+1 for i in np.nonzero(np.diff(kvals)<0)[0]]
    index_once = np.arange(ends[0])
    if index:
        return index_once
    else:
        return kvals[index_once]

# Get the index from the full k_array of only one of the multipoles
def get_l_index(il,kvals):
	ends = [len(kvals)]
	#print ends

	ends[:0] = [i+1 for i in np.nonzero(np.diff(kvals)<0)[0]]
	ends [:0] = [0]
	#print ends
	#print ends[il+1]

	return np.arange(ends[il],ends[il+1])

# Get the boolean index from the full k_array of only one of the multipoles
def get_l_boolindex(il,kvals):
	
	index = get_l_index(il,kvals)
	n = len(kvals)

	boolindex = np.zeros(n,dtype=int)
	boolindex[index] = 1

	return boolindex

def float_unique(arr,TOL=-15):
	""" Uses the numpy.unique function to find unique floating points (i.e., ones that differ
		less than the specified log10 tolerance, TOL). Note that things close to the tolerance can still
		get lost, so you should choose something far from the true differences in your array, 
		and far from the numerical precision of your system. The result will only be accurate
		to the level specified by TOL! """

	# Round to TOL order of magnitude
	O = 10.**TOL
	arr = O*np.round(arr/O,0)

	# Return a sorted array of unique floating point values
	# Note that floaring point errors will cause some of the entries to
	# appear more than once. We need to correct this.
	arr = np.unique(arr) 

	return arr

if __name__=='__main__':

	# Verification tests are below: the shapes of the arrays are what we expect
	# Validation tests: compare to other covmats (e.g. that calculated from Patchy mocks)

	np.set_printoptions(precision=2,suppress=True,linewidth=150)

	print('Legendre',legendre(0)(1),'\n')
	
	kvals = np.arange(4)*2e-2 + 2e-2
	pk1d = 33000.*np.arange(len(kvals)*3)+3000.01

	kvals,muvals,pk2d = pkto2d(kvals,pk1d,[0,2,4],Nmu=10000)
	print('nk =',kvals.shape, 'nmu =',muvals.shape, 'pk2d shape =',pk2d.shape, '\n')
	print('kold =', kvals)

	kvs,pk1dnew = pkto1d(kvals,muvals,pk2d)
	print('old =', pk1d)
	print('new =', pk1dnew)
	print('knew =', kvs)
	print('')

	cm = covll(kvals,muvals,pk2d,1.e18,1.)
	print('covmat shape =', cm.shape, '(3 multipoles)\n')
	print('covmat =\n',cm)

	il = 1
	print('\n', kvs[get_l_index(il,kvs)], '\n', get_k_once(kvs))
	print('\n', pk1dnew[get_l_index(il,kvs)], '\n', pk1dnew)
	print(get_l_boolindex(il,kvs))
