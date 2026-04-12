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

eb_typ_dict={'names': ('name', 'sptype1', 'sptype2', 'per', 'Vmag', 'BmV', 'logm1', 'logm1_err', 'logm2', 'logm2_err', 'logr1', 'logr1_err', 'logr2', 'logr2_err',
'logg1', 'logg1e', 'logg2', 'logg2e', 'logteff1', 'logteff1_err', 'logteff2', 'logteff2_err', 'logL1', 'logL1e', 'logL2', 'logL2e', 'MoH', 'MoHe'), 
'formats': ('U32', 'str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 
'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')}

#eb_p,eb_m1,eb_m1_err,eb_r1,eb_r1_err,eb_teff1,eb_teff1_err, eb_m2,eb_m2_err,eb_r2,eb_r2_err,eb_teff2,eb_teff2_err 
ebs = np.loadtxt("debcat_20260411.dat", unpack=True, comments='#', delimiter=' ',dtype=eb_typ_dict)#
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
#	eb_rinf_err[i] = math.sqrt(pow(eb
	   
print("DEBCat M-dwarf names, periods, rinf, tinf, and linf:")
for i in range(len(eb_p)):
	print(eb_names[i], eb_p[i], eb_rinf[i], "+/-", eb_rinf_err[i], eb_teffinf[i], "+/-", eb_teffinf_err[i], eb_linf[i], "+/-", eb_linf_err[i])

np.savetxt('output.out',[(eb_names[i], eb_p[i], eb_rinf[i], eb_rinf_err[i], eb_teffinf[i], eb_teffinf_err[i], eb_linf[i], eb_linf_err[i]) for i in range(len(eb_p))], delimiter='\t', fmt=('%s'), header='name per rinf rinf_err tinf tinf_err linf linf_err')

fig = plt.figure(figsize=(5,3))
plt.errorbar(eb_p,np.array(eb_rinf),yerr=eb_rinf_err,linestyle='',color='grey')
plt.scatter(eb_p,np.array(eb_rinf),color='grey',label=r'${\rm DEBCat\ M\ Dwarfs}$')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0.1,100])
plt.ylim([0.9,1.3])
plt.legend(loc='upper left')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'$R_{\rm obs}/R_{\rm model}$',fontsize=16)
plt.xscale('log')
plt.tight_layout()
plt.savefig("rinf_debcat.pdf")
plt.savefig("rinf_debcat.png", dpi=1200)


fig2 = plt.figure(figsize=(5,3))
plt.errorbar(eb_p,eb_teffinf,yerr=eb_teffinf_err,linestyle='',color='grey')
plt.scatter(eb_p,eb_teffinf,color='grey',label='DEBCat M Dwarfs')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0.1,100])
plt.legend(loc='best')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'$T_{\rm eff,\ obs}/T_{\rm eff,\ model}$',fontsize=16)
plt.xscale('log')
plt.tight_layout()
plt.savefig('tinf_debcat.pdf')
plt.savefig('tinf_debcat.png', dpi=1200)

fig25 = plt.figure(figsize=(5,3))
plt.errorbar(eb_p,eb_linf,yerr=eb_linf_err,linestyle='',color='grey')
plt.scatter(eb_p,eb_linf,color='grey',label='DEBCat M Dwarfs')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0.1,100])
plt.legend(loc='best')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'$L_{\rm obs}/L_{\rm model}$',fontsize=16)
plt.xscale('log')
plt.tight_layout()
plt.ylim(0.5, 1.5)
plt.savefig('linf_debcat.pdf')
plt.savefig('linf_debcat.png', dpi=1200)

fig3 = plt.figure(figsize=(5,3))
plt.scatter(eb_p,eb_m,color='grey',label='DLEB M Dwarfs')
plt.axvline(6.5,color='k',linestyle='dotted')
plt.xlabel(r'$\rm Period\ (days)$',fontsize=16)
plt.ylabel(r'${\rm Mass\ (M_{\odot})}$',fontsize=16)
plt.xscale('log')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('mass_comp_debcat.pdf')

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
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0,0.8])
plt.legend(loc='best')
plt.xlabel(r'$\rm Mass\ (M_{\odot})$',fontsize=16)
plt.ylabel(r'$R_{\rm obs}/R_{\rm model}$',fontsize=16)
plt.tight_layout()
plt.savefig("rinf_mass_debcat.pdf")
plt.savefig("rinf_mass_debcat.png",dpi=1200)

fig = plt.figure()
plt.errorbar(eb_m,eb_teffinf,xerr=eb_m_err,yerr=eb_teffinf_err,linestyle='',color='grey')
plt.scatter(eb_m,eb_teffinf,color='grey',label='DLEB M Dwarfs')
plt.axhline(1.0,linestyle='--',color='k')
plt.xlim([0,0.8])
plt.legend(loc='best')
plt.xlabel(r'$\rm Mass\ (M_{\odot})$',fontsize=16)
plt.ylabel(r'$T_{\rm eff,\ obs}/T_{\rm eff,\ model}$',fontsize=16)
plt.tight_layout()
plt.savefig("tinf_mass.pdf")
plt.savefig("tinf_mass.png",dpi=1200)
