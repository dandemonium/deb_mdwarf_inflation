###################################################
###	Compares measured M,R,Teff to isochrones, ###
###	plots their ratios, and compares against ###
###	literature DLEB M-dwarfs.		###
###################################################
### Import modules ###
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from astropy import constants as u
from scipy import stats,optimize
from astropy.modeling import models, fitting
import pandas

teffsun = 5777 #from BOB

kelpebfile = sys.argv[1] ##file containing KELP M-dwarf properties - #name period(days) age(gyr) m2(msun) m2_err(msun) r2(rsun) r2_err(rsun) teff2(K) teff2_err(K)
#eb_typ_dict={'names': ('name', 'sptype1', 'sptype2', 'per', 'Vmag', 'BmV', 'logm1', 'logm1_err', 'logm2', 'logm2_err', 'logr1', 'logr1_err', 'logr2', 'logr2_err', 'logg1', 'logg1e', 'logg2', 'logg2e', 'logteff1', 'logteff1_err', 'logteff2', 'logteff2_err', 'logL1', 'logL1e', 'logL2', 'logL2e', 'MoH', 'MoHe'), 'formats': ('U30', 'str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')}

#eb_p,eb_m1,eb_m1_err,eb_r1,eb_r1_err,eb_teff1,eb_teff1_err, eb_m2,eb_m2_err,eb_r2,eb_r2_err,eb_teff2,eb_teff2_err 
ebs = np.genfromtxt("debcat_20260411.dat", unpack=True, comments='#', delimiter=' ',dtype=None, encoding='utf-8')
#ebs = np.loadtxt("debcat_20260411.dat", unpack=True, comments='#', delimiter=' ',dtype=eb_typ_dict)#
#eb_p,eb_m,eb_m_err,eb_r,eb_r_err,eb_teff,eb_teff_err = np.loadtxt("known_MMBinaries.txt",unpack=True,usecols=(0,1,2,3,4,5,6),comments='#')

# keep only stars w/ M < 0.6 Msun:
m1dx = np.where(ebs[6] <= np.log10(0.6))
m2dx = np.where(ebs[8] <= np.log10(0.6))
eb_names_A = [i + "_A" for i in ebs[0][m1dx]]
eb_names_B = [i + "_B" for i in ebs[0][m2dx]]

eb_names = np.concatenate((eb_names_A, eb_names_B), axis=None)
eb_logm = np.concatenate((ebs[6][m1dx], ebs[8][m2dx]), axis=None)
eb_logm_err = np.concatenate((ebs[7][m1dx], ebs[9][m2dx]), axis=None)
eb_logr = np.concatenate((ebs[10][m1dx], ebs[12][m2dx]), axis=None)
eb_logr_err = np.concatenate((ebs[11][m1dx], ebs[13][m2dx]), axis=None)
eb_logteff = np.concatenate((ebs[18][m1dx], ebs[20][m2dx]), axis=None)
eb_logteff_err = np.concatenate((ebs[19][m1dx], ebs[21][m2dx]), axis=None)
eb_loglum = np.concatenate((ebs[22][m1dx], ebs[24][m2dx]), axis=None)
eb_loglum_err = np.concatenate((ebs[23][m1dx], ebs[25][m2dx]), axis=None)
eb_p = np.concatenate((ebs[3][m1dx], ebs[3][m2dx]), axis=None)
eb_m = 10.**eb_logm
eb_m_err = np.log(10) * eb_m * eb_logm_err
eb_r = 10.**eb_logr
eb_r_err = np.log(10) * eb_r * eb_logr_err
eb_teff = 10.**eb_logteff
eb_teff_err = np.log(10) * eb_teff * eb_logteff_err
eb_lum = 10.**eb_loglum
eb_lum_err = np.log(10) * eb_lum * eb_loglum_err

dtyp_dict={'names': ('ticid', 'name','period', 'age', 'm2', 'm2_err', 'r2', 'r2_err', 'teff2', 'teff2_err', 'notes'), 'formats': ('str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'str')}
#dtyp_dict=('S15', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')
#name,kelt_p,age,kelt_m,kelt_m_err,kelt_r,kelt_r_err,kelt_teff,kelt_tefferr = np.loadtxt(ebfile,dtype=dtyp_dict,comments="#",usecols=(0,1,2,3,4,5,6,7))
ticid, name, kelt_p, kelt_age, kelt_m, kelt_m_err, kelt_r, kelt_r_err, kelt_teff, kelt_teff_err, notes = np.genfromtxt(kelpebfile, unpack=True, dtype=None, encoding='utf-8')
#np.loadtxt(kelpebfile,unpack=True, dtype=dtyp_dict, delimiter='\t')#np.genfromtxt(ebfile,dtype=dtyp_dict,unpack=True,comments="#")
iso_m_80m,iso_teff_80m,iso_r_80m = np.loadtxt("b15_80myr.dat",unpack=True)
mrt_spl_80m = CubicSpline(iso_m_80m,[(iso_r_80m[i],iso_teff_80m[i]) for i in range(len(iso_teff_80m))])
iso_m_120m,iso_teff_120m,iso_r_120m = np.loadtxt("b15_120myr.dat",unpack=True)
mrt_spl_120m = CubicSpline(iso_m_120m,[(iso_r_120m[i],iso_teff_120m[i]) for i in range(len(iso_teff_120m))])
iso_m_200m,iso_teff_200m,iso_r_200m = np.loadtxt("b15_200myr.dat",unpack=True)
mrt_spl_200m = CubicSpline(iso_m_200m,[(iso_r_200m[i],iso_teff_200m[i]) for i in range(len(iso_teff_200m))])
iso_m_300m,iso_teff_300m,iso_r_300m = np.loadtxt("b15_300myr.dat",unpack=True)
mrt_spl_300m = CubicSpline(iso_m_300m,[(iso_r_300m[i],iso_teff_300m[i]) for i in range(len(iso_teff_300m))])
iso_m_400m,iso_teff_400m,iso_r_400m = np.loadtxt("b15_400myr.dat",unpack=True)
mrt_spl_400m = CubicSpline(iso_m_400m,[(iso_r_400m[i],iso_teff_400m[i]) for i in range(len(iso_teff_400m))])
iso_m_1g,iso_teff_1g,iso_r_1g = np.loadtxt("b15_1gyr.dat",unpack=True)
mrt_spl_1g = CubicSpline(iso_m_1g,[(iso_r_1g[i],iso_teff_1g[i]) for i in range(len(iso_teff_1g))])
iso_m_2g,iso_teff_2g,iso_r_2g = np.loadtxt("b15_2gyr.dat",unpack=True)
mrt_spl_2g = CubicSpline(iso_m_2g,[(iso_r_2g[i],iso_teff_2g[i]) for i in range(len(iso_teff_2g))])
iso_m_5g,iso_teff_5g,iso_r_5g = np.loadtxt("b15_5gyr.dat",unpack=True)
mrt_spl_5g = CubicSpline(iso_m_5g,[(iso_r_5g[i],iso_teff_5g[i]) for i in range(len(iso_teff_5g))])
iso_m_10g,iso_teff_10g,iso_r_10g = np.loadtxt("b15_10gyr.dat",unpack=True)
mrt_spl_10g = CubicSpline(iso_m_10g,[(iso_r_10g[i],iso_teff_10g[i]) for i in range(len(iso_teff_10g))])

kelt_r_mod = np.zeros(len(kelt_m))
kelt_r_mod_err = np.zeros(len(kelt_m))
kelt_rinf = np.zeros(len(kelt_m))
kelt_rinf_err = np.zeros(len(kelt_m))

kelt_teff_mod = np.zeros(len(kelt_m))
kelt_teff_mod_err = np.zeros(len(kelt_m))
kelt_teffinf = np.zeros(len(kelt_m))
kelt_teffinf_err = np.zeros(len(kelt_m))
for i in range(len(kelt_m)):
	kelt_mass_draws = np.random.normal(kelt_m[i],kelt_m_err[i],50000)
	tmp_age = kelt_age[i]

	if tmp_age < 0.1: #use 80 Myr iso
		spl = mrt_spl_80m
	elif 0.1 <= tmp_age < 0.16: # use 120 Myr iso
		spl = mrt_spl_120m
	elif 0.16 <= tmp_age < 0.25: # use 200 Myr iso
		spl = mrt_spl_200m
	elif 0.25 <= tmp_age < 0.35: # use 300 Myr iso
		spl = mrt_spl_300m
	elif 0.35 <= tmp_age < 0.7: # use 400 Myr iso
		spl = mrt_spl_400m
	elif 0.7 <= tmp_age < 1.5: # use 1 Gyr iso
		spl = mrt_spl_1g
	elif 1.5 <= tmp_age < 3.5: # use 2 Gyr iso
		spl = mrt_spl_2g
	elif 3.5 <= tmp_age < 7.5: # use 5 Gyr iso
		spl = mrt_spl_5g
	elif 7.5 <= tmp_age: # use 10 Gyr iso
		spl = mrt_spl_10g

	kelt_r_mod[i], kelt_teff_mod[i] = spl(kelt_m[i])
	tmp =  spl(kelt_mass_draws)
	kelt_r_mod_err[i] = np.std([j[0] for j in tmp])
	kelt_teff_mod_err[i] = np.std([j[1] for j in tmp])
	kelt_rinf[i] = kelt_r[i]/kelt_r_mod[i]
	kelt_rinf_err[i] = kelt_rinf[i] * np.sqrt((kelt_r_err[i]/kelt_r[i])**2. + (kelt_r_mod_err[i]/kelt_r_mod[i])**2.)
	kelt_teffinf[i] = kelt_teff[i]/kelt_teff_mod[i]
	kelt_teffinf_err[i] = kelt_teffinf[i] * np.sqrt((kelt_teff_err[i]/kelt_teff[i])**2. + (kelt_teff_mod_err[i]/kelt_teff_mod[i])**2.)

	kelt_rinf[i] -= 1.0
	kelt_teffinf[i] -= 1.0
	#kelt_linf[i] -= 1.0
	

eb_r_mod = np.zeros(len(eb_m))
eb_r_mod_err = np.zeros(len(eb_m))
eb_rinf = np.zeros(len(eb_m))
eb_rinf_err = np.zeros(len(eb_m))
eb_teff_mod = np.zeros(len(eb_m))
eb_teff_mod_err = np.zeros(len(eb_m))
eb_teffinf = np.zeros(len(eb_m))
eb_teffinf_err = np.zeros(len(eb_m))
eb_l = eb_lum
#eb_l_err = np.sqrt((2. * eb_r * eb_r_err)**2. + (4.*(eb_teff/teffsun)**3. * eb_teff_err/teffsun)**2.)
eb_l_err = eb_lum_err
eb_l_mod = np.zeros(len(eb_l))
eb_l_mod_err = np.zeros(len(eb_l))
eb_linf = np.zeros(len(eb_l))
eb_linf_err = np.zeros(len(eb_l))
for i in range(len(eb_m)): # use 2 Gyr iso for no good reason
    if eb_names[i] == 'Gaia_DR3_6751389835685124992': #'M55 V44 is ancient, per Kaluzny et al. 2014, AcA, 64, 11K
    	eb_mass_draws = np.random.normal(eb_m[i],eb_m_err[i],5000)
    	eb_r_mod[i], eb_teff_mod[i] = mrt_spl_10g(eb_m[i])
    	tmp = mrt_spl_10g(eb_mass_draws)
    else:
    	eb_mass_draws = np.random.normal(eb_m[i],eb_m_err[i],5000)
    	eb_r_mod[i], eb_teff_mod[i] = mrt_spl_1g(eb_m[i])
    	tmp = mrt_spl_1g(eb_mass_draws)    
    mri_draws = [j[0] for j in tmp]
    mti_draws = [j[1] for j in tmp]
    eb_r_mod_err[i] = np.std(mri_draws)
    eb_teff_mod_err[i] = np.std(mti_draws)
    eb_rinf[i] = eb_r[i]/eb_r_mod[i]
    eb_rinf_err[i] = eb_rinf[i] * np.sqrt((eb_r_err[i]/eb_r[i])**2. + (eb_r_mod_err[i]/eb_r_mod[i])**2.)
    eb_teffinf[i] = eb_teff[i]/eb_teff_mod[i]
    eb_teffinf_err[i] = eb_teffinf[i] * np.sqrt((eb_teff_err[i]/eb_teff[i])**2. + (eb_teff_mod_err[i]/eb_teff_mod[i])**2.)
    eb_l_mod[i] = (eb_r_mod[i]**2.) * (eb_teff_mod[i]/teffsun)**4.
    eb_l_mod_err[i] = eb_l_mod[i] * np.sqrt( (2.*eb_r_mod_err[i]/eb_r_mod[i])**2. + (4.*eb_teff_mod_err[i]/eb_teff_mod[i])**2.)
    if eb_loglum[i] < -8.:
        eb_linf[i] = (eb_r[i]**2. * (eb_teff[i]/teffsun)**4.) / eb_l_mod[i]
    else:
        eb_linf[i] = eb_lum[i] / eb_l_mod[i]
    eb_linf_err[i] = eb_linf[i] * np.sqrt( (2.*eb_r_err[i]/eb_r[i])**2. + (4.*eb_teff_err[i]/eb_teff[i])**2. + (eb_l_mod_err[i]/eb_l_mod[i])**2. )   
    eb_rinf[i] -= 1.0
    eb_teffinf[i] -= 1.0
    eb_linf[i] -= 1.0
print("DEBCat M-dwarf names, periods, rinf, tinf, and linf:")
for i in range(len(eb_p)):
	print(eb_names[i], eb_p[i], eb_rinf[i], "+/-", eb_rinf_err[i], eb_teffinf[i], "+/-", eb_teffinf_err[i], eb_linf[i], "+/-", eb_linf_err[i])

np.savetxt('output.out',[(eb_names[i], eb_p[i], eb_rinf[i], eb_rinf_err[i], eb_teffinf[i], eb_teffinf_err[i], eb_linf[i], eb_linf_err[i]) for i in range(len(eb_p))], delimiter='\t', fmt=('%s'), header='name per rinf rinf_err tinf tinf_err linf linf_err')
print("\n")
print("----------")
print("KELP M-dwarf obs. and model radii:", kelt_r, kelt_r_mod)
print("\n")
print("KELP M-dwarf obs. and model Teff:",kelt_teff, kelt_teff_mod)
print("\n")
print("KELP M-dwarf model radius error:",kelt_r_mod_err)
print("\n")
print("KELP: ", len(ticid), " systems")
print("Period Range (days):", kelt_p.min(), "--", kelt_p.max())
print("Mass Range (Msun):", kelt_m.min(), "--", kelt_m.max())
print("Radius Range (Rsun):", kelt_r.min(), "--", kelt_r.max())
print("Teff Range (K):", kelt_teff.min(), "--", kelt_teff.max())
print("Age Range (Gyr):", kelt_age.min(), "--", kelt_age.max())
print("\n")
print("TIC ID", "\t\t", "Name", "\t\t", "Rad. inf. (O-C; %)", "\t", "Rad. inf. err. (%)", "\t", "Teff. supp. (O-C; %)", "\t", "Teff. supp. err (%)")
for i in range(len(kelt_rinf)): print("TIC " + str(ticid[i]), "\t", str(name[i]), "\t", kelt_rinf[i]*100., "\t", kelt_rinf_err[i]*100., "\t", 
										kelt_teffinf[i]*100., "\t", kelt_teffinf_err[i]*100.)
print("\n")
print("KELP:", "\t\t", "Rad. inf. (O-C; %)", "\t", "Rad. inf. err. (%)", "\t", "Teff. supp. (O-C; %)", "\t", "Teff. supp. err (%)")
print("Average", "\t", np.nanmean(kelt_rinf)*100., "\t", np.nanstd(kelt_rinf)*100., "\t", 
										np.nanmean(kelt_teffinf)*100., "\t", np.nanstd(kelt_teffinf)*100.)
print("Average", "\t", np.nanmean(kelt_rinf)*100., "\t", np.mean(kelt_rinf_err)*100., "\t", 
										np.nanmean(kelt_teffinf)*100., "\t", np.nanmean(kelt_teffinf_err)*100.)
#tot_r_inf = np.array([list(eb_rinf)+list(kelt_rinf)])
#tot_teffinf = np.array([list(eb_teffinf)+list(kelt_teffinf)])
#tot_p = np.array([list(eb_p) + list(kelt_p)])
#tot_teffinf_err = np.array([list(eb_teffinf_err)+list(kelt_teffinf_err)])
print("\n")
#print("r-coefficient for radius inflation:",np.corrcoef(tot_r_inf,np.log(tot_p)))
#print("rho-coefficient for radius inflation:",stats.spearmanr(tot_r_inf,np.log(tot_p)))

#print("r-coefficient for teff deflation:",np.corrcoef(tot_teffinf,np.log(tot_p)))
#print("rho-coefficient for teff deflation:",stats.spearmanr(tot_teffinf,np.log(tot_p)))
#print("Average temperature inflation among DLEB M dwarfs:",np.mean(tot_teffinf)

### Fit a line to the data
# initialize a linear fitter
#fit = fitting.LinearLSQFitter()

# initialize a linear model
#line_init = models.Linear1D()

# fit the Teff inflation with the fitter
#fitted_line_teffs = fit(line_init, np.log(tot_p), tot_teffinf, weights=1.0/tot_teffinf_err)

#line = lambda x, m, b: b + m*np.log(x)
#eb_topt, eb_tcov = optimize.curve_fit(line,eb_p,eb_teffinf,sigma=eb_teffinf_err,absolute_sigma=True)
#eb_teffinf_curve_errs = np.sqrt(np.diagonal(eb_tcov))
#print("Teff inf slope and intercept, and errors", popt, errs)
#eb_teffline = line(np.sort(eb_p),*(popt))
#eb_upper_teffline = line(np.sort(eb_p),*(popt+errs))
#eb_lower_teffline = line(np.sort(eb_p),*(popt-errs))
#eb_upper3sig_teffline = line(np.sort(eb_p),*(popt+3.*errs))
#eb_lower3sig_teffline = line(np.sort(eb_p),*(popt-3.*errs))

### Fit Radius inflation w/ the fitter
#eb_ropt, eb_rcov = optimize.curve_fit(line,eb_p,eb_rinf,sigma=eb_rinf_err,absolute_sigma=True)
#eb_rinf_curve_errs = np.sqrt(np.diagonal(eb_rcov))
#print("Rad inf slope and intercept, and errors",eb_ropt, eb_rinf_errs)
#eb_radline = line(np.sort(eb_p),*(eb_ropt))
#eb_upper_radline = line(np.sort(eb_p),*(eb_ropt+eb_rinf_errs))
#eb_lower_radline = line(np.sort(eb_p),*(eb_ropt-eb_rinf_errs))
#eb_upper3sig_radline = line(np.sort(eb_p),*(eb_ropt+3.*eb_rinf_errs))
#eb_lower3sig_radline = line(np.sort(eb_p),*(eb_ropt-3.*eb_rinf_errs))


###### Plot it all ######
deb_mkr = 'o'
deb_clr = 'grey'
deb_mkreclr = deb_clr
deb_fl = 'full' #'none'
deb_msz = 4
deb_a = 1#0.5
deb_lbl = r'${\rm DEBCat\ M\ Dwarfs}$'
	
sleb_mkr = 'o'
sleb_clr ='r'
sleb_mkreclr = 'r'
sleb_fl = 'full'
sleb_msz = deb_msz
sleb_a = 1#0.75
sleb_lbl = r'${\rm SLEB\ sample}$'

lbl_fsz = 14

fig = plt.figure(figsize=(5,3))
plt.errorbar(eb_p, eb_rinf*100., yerr=eb_rinf_err*100., linestyle='', color=deb_clr, marker=deb_mkr, markersize=deb_msz, fillstyle=deb_fl, label=deb_lbl, alpha=deb_a)

plt.errorbar(kelt_p,kelt_rinf*100., yerr=kelt_rinf_err*100., 
	linestyle='', color=sleb_clr, markeredgecolor=sleb_mkreclr, marker=sleb_mkr, markersize=sleb_msz, fillstyle=sleb_fl, label=sleb_lbl, alpha=sleb_a, zorder=20)
#plt.axvline(6.5,color='k',linestyle='dotted')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim(0.4,100)
plt.ylim(-10,25)
plt.legend(loc='upper left')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=lbl_fsz)
plt.ylabel(r'$\frac{\Delta R}{R} (\%)$', fontsize=lbl_fsz)
#plt.ylabel(r'$R_{\rm obs}/R_{\rm model}$',fontsize=16)
#plt.ylabel(r'$(R_{\rm obs} - R_{\rm mod}$/R_{\rm mod} (\%)$',fontsize=lbl_fsz)
plt.xscale('log')
plt.tight_layout()
plt.savefig("rinf.pdf")
plt.savefig("rinf.png", dpi=1200)
#plt.show()


fig2 = plt.figure(figsize=(5,3))
plt.errorbar(eb_p, eb_teffinf*100., yerr=eb_teffinf_err*100.,
	linestyle='', color=deb_clr, marker=deb_mkr, markeredgecolor=deb_mkreclr, markersize=deb_msz, fillstyle=deb_fl, label=deb_lbl, alpha=deb_a)
plt.errorbar(kelt_p, kelt_teffinf*100., yerr=kelt_teffinf_err*100., 
	linestyle='', color=sleb_clr, markeredgecolor=sleb_mkreclr, marker=sleb_mkr, markersize=sleb_msz, fillstyle=sleb_fl, label=sleb_lbl, alpha=sleb_a)

#plt.axvline(6.5,color='k',linestyle='dotted')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim(0.4,100)
plt.ylim(-20, 20)
plt.legend(loc='upper left')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=lbl_fsz)
plt.ylabel(r'$\frac{\Delta T_{\rm eff}}{T_{\rm eff}} (\%)$',fontsize=lbl_fsz)
#plt.ylabel(r'$T_{\rm eff,\ obs}/T_{\rm eff,\ model}$',fontsize=16)
#plt.ylabel(r'$(T_{\rm eff,obs} - T_{\rm eff,mod})/T_{\rm eff,mod} (\%)$',fontsize=16)
plt.xscale('log')
plt.tight_layout()
plt.savefig('tinf.pdf')
plt.savefig('tinf.png', dpi=1200)

fig25 = plt.figure(figsize=(5,3))
plt.errorbar(eb_p,eb_linf,yerr=eb_linf_err,linestyle='',color='grey')
#plt.axvline(6.5,color='k',linestyle='dotted')
plt.scatter(eb_p,eb_linf,color='grey',label='DEBCat M Dwarfs')
#plt.plot(np.sort(eb_p),teffline)
#plt.fill_between(np.sort(eb_p),lower3sig_teffline, upper3sig_teffline, color='tab:red',alpha=0.25)
#plt.fill_between(np.sort(eb_p),lower_teffline, upper_teffline, color='tab:red',alpha=0.25)
#plt.errorbar(kelt_p,kelt_teffinf,yerr=kelt_teffinf_err,linestyle='',color='r',zorder=20)
#plt.scatter(kelt_p,kelt_teffinf,color='r',label=r'${\rm TOI-2065 B}$',zorder=20)
#plt.errorbar(kelt_p[0], kelt_teffinf[0], color='r', linestyle='',yerr=kelt_teffinf_err[0], zorder=20)
#plt.scatter(kelt_p[0], kelt_teffinf[0], color='r', label=r'${\rm HD\ 74925\ B\ (NextGen\ SED)}$', marker='o', zorder=20)#, label=r'${\rm our\ sample}$',marker='o',zorder=20)#,s=5)
#plt.errorbar(kelt_p[1],kelt_teffinf[1],yerr=kelt_teffinf_err[1],linestyle='',color='tab:red',zorder=15)
#plt.scatter(kelt_p[1],kelt_teffinf[1],color='tab:red',label=r'${\rm HD\ 74925\ B\ (MIST\ iso.)}$',zorder=15)#,s=5)
#plt.errorbar(kelt_p[1],kelt_teffinf[1],yerr=kelt_teffinf_err[1],linestyle='',color='tab:red',zorder=20)
#plt.scatter(kelt_p[1],kelt_teffinf[1],color='tab:red',label=r'${\rm With-MIST\ Fit}$',zorder=20)
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0.1,100])
plt.legend(loc='best')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'$L_{\rm obs}/L_{\rm model}$',fontsize=16)
plt.xscale('log')
plt.tight_layout()
plt.ylim(0.5, 1.5)
plt.savefig('linf.pdf')
plt.savefig('linf.png',dpi=1200)

fig3 = plt.figure(figsize=(5,3))
plt.scatter(eb_p,eb_m,color='grey',label='DLEB M Dwarfs')
plt.axvline(6.5,color='k',linestyle='dotted')
#plt.scatter(kelt_p,kelt_m,label=r'${\rm our\ sample}$',color='r',zorder=20)
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'${\rm Mass\ (M_{\odot})}$',fontsize=16)
plt.xscale('log')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('mass_comp.pdf')

fig = plt.figure()
conv_ndx = np.where(eb_m <= 0.3)
fully_conv_p = eb_p[conv_ndx]
fully_conv_m = eb_m[conv_ndx]
fully_conv_r = eb_r[conv_ndx]
part_ndx = np.where(eb_m > 0.3)
partially_conv_p = eb_p[part_ndx]
partially_conv_m = eb_m[part_ndx]
partially_conv_r = eb_r[part_ndx]
fully_conv_rinf_err = eb_rinf_err[conv_ndx]
partially_conv_rinf_err = eb_rinf_err[part_ndx]

plt.scatter(partially_conv_p,partially_conv_rinf_err*100.,marker='x',color='blue',label=r'$M > 0.3 M_{\odot}$')
plt.scatter(fully_conv_p,fully_conv_rinf_err*100.,marker='o',color='red',label=r'$M \leq 0.3 M_{\odot}$')
plt.axvline(6.5,color='k',linestyle='dotted')
plt.axhline(5,color='k',linestyle='dashed')
plt.xlabel('Orbital Period (days)')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Uncertainty on Radius Inflation (%)')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('rinf_err_vs_period_debcat.pdf')

fig = plt.figure()
plt.scatter(partially_conv_m,partially_conv_r,marker='x',color='blue',label=r'$M > 0.3 M_{\odot}$')
plt.scatter(fully_conv_m,fully_conv_r,marker='o',color='red',label=r'$M \leq 0.3 M_{\odot}$')
plt.plot(iso_m_1g,iso_r_1g,color='k',label=r'$1\ {\rm Gyr}$') 
plt.plot(iso_m_5g,iso_r_5g,color='k',label=r'$5\ {\rm Gyr}$',linestyle='dotted') 
plt.xlabel(r'${\rm Mass\ (M_{\odot})}$')
plt.ylabel(r'${\rm Radius\ (R_{\odot})}$')
plt.xlim([0.08,0.75])
plt.ylim([0.08,0.75])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('debcat_mass_radius.pdf')

fig = plt.figure()
plt.errorbar(eb_m,np.array(eb_rinf),xerr=eb_m_err,yerr=eb_rinf_err,linestyle='',color='grey')
plt.scatter(eb_m,np.array(eb_rinf),color='grey',label=r'${\rm DLEB\ M\ Dwarfs}$')
#plt.errorbar(kelt_m,kelt_rinf,color='r',linestyle='',xerr=kelt_m_err,yerr=kelt_rinf_err,zorder=20)
#plt.scatter(kelt_m,kelt_rinf,color='r',label=r'${\rm our\ sample}$',marker='o',zorder=20)
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0,0.8])
#plt.ylim([0.8,1.5])
plt.legend(loc='best')
plt.xlabel(r'$\rm Mass\ (M_{\odot})$',fontsize=16)
plt.ylabel(r'$R_{\rm obs}/R_{\rm model}$',fontsize=16)

#plt.yscale('log')
plt.tight_layout()
plt.savefig("rinf_mass.pdf")

fig = plt.figure()
plt.errorbar(eb_m,eb_teffinf,xerr=eb_m_err,yerr=eb_teffinf_err,linestyle='',color='grey')
plt.scatter(eb_m,eb_teffinf,color='grey',label='DLEB M Dwarfs')
#plt.errorbar(kelt_m,kelt_teffinf,xerr=kelt_m_err,yerr=kelt_teffinf_err,linestyle='',color='r',zorder=20)
#plt.scatter(kelt_m,kelt_teffinf,color='r',label=r'${\rm our\ sample}$',zorder=20)
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0,0.8])
#plt.ylim([0.8,1.5])
plt.legend(loc='best')
plt.xlabel(r'$\rm Mass\ (M_{\odot})$',fontsize=16)
plt.ylabel(r'$T_{\rm eff,\ obs}/T_{\rm eff,\ model}$',fontsize=16)

#plt.yscale('log')
plt.tight_layout()
plt.savefig("tinf_mass.pdf")
