# A list of kernels used to displace particles
# Add a sea-ice kernel https://github.com/AnnekeV/thesis_plastic_seaice/blob/master/kernels.py

from parcels import ParcelsRandom, StatusCode
import math

# ERA 5 https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation


def StokesDrift(particle, fieldset, time):
    """Stokes drift kernel.

    Description
    ----------
    Using the approach in [1] assuming a Phillips wave spectrum to determine
    the depth dependent Stokes drift. Specifically, the 'Stokes drift velocity'
    :math:`u_s` is computed as per Eq. (19) in [1].

    We treat the Stokes drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_s * u_s(x,t)`
    where :math:`u_c` is the current velocity, :math:`u_s` is the Stokes drift velocity,
    and :math:`C_s` is the depth-varying decay factor.

    For further description, see https://plastic.parcels-code.org/en/latest/physicskernels.html#stokes-drift

    Parameter Requirements
    ----------
    fieldset :
        - `fieldset.Stokes_U` and `fieldset.Stokes_V`, the Stokes drift velocity fields. Units [m s-1]
        - `fieldset.wave_Tp`, the peak wave period field (:math:`T_p`). Units [s].

    Kernel Requirements
    ----------
    Order of Operations:
        None - can be applied at any time.

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005

    """
    # Sample the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    # Compute the local bathymetry / water depth with a margin of error
    local_bathymetry = 0.99*fieldset.bathymetry[time, particle.depth, particle.lat, particle.lon]

    # Only compute displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14 and particle.depth < local_bathymetry:
        # Sample the U / V components of Stokes drift
        stokes_U = fieldset.Stokes_U[time, particle.depth, particle.lat, particle.lon]
        stokes_V = fieldset.Stokes_V[time, particle.depth, particle.lat, particle.lon]

        # Peak wave frequency
        omega_p = 2. * math.pi / T_p

        # Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        # Repeated inner term of Eq. (19) - note depth is negative in this formulation, but model depths are positive by convention
        kp_z_2 = 2. * k_p * particle.depth

        # Decay factor in Eq. (19) -- Where beta=1 for the Phillips spectrum
        decay = math.exp(-kp_z_2) - math.sqrt(math.pi * kp_z_2) * math.erfc(math.sqrt(kp_z_2))

        # Apply Eq. (19) and compute particle displacement
        particle_dlon += stokes_U * decay * particle.dt  # noqa
        particle_dlat += stokes_V * decay * particle.dt  # noqa


# Wind related kernels
def WindageDrift(particle, fieldset, time):
    """Leeway windage kernel.

    Description
    ----------
    A simple windage kernel that applies a linear relative 'wind velocity'
    to the particle.

    We treat the windage drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_w * (u_w(x,t)-u_c(x,t))`
    where :math:`u_c` is the ocean current velocity, :math:`u_w` is the wind velocity
    at 10m height, and :math:`C_w` is the windage coefficient (usually taken to
    be in [1%,5%], depending on particle size)

    For further description, see https://plastic.parcels-code.org/en/latest/physicskernels.html#wind-induced-drift-leeway

    Parameter Requirements
    ----------
    particle :
        - wind_coefficient - the particle windage coefficient in decimals (usually taken to be between 0.01 and 0.05, depending on particle size).
    fieldset :
        - `fieldset.Wind_U` and `fieldset.Wind_V`, the wind velocity field at 10m height above sea surface. Units [m s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        None - can be applied at any time.

    """
    # Sample ocean velocities
    (ocean_U, ocean_V) = fieldset.UV[particle]
    ocean_speed = math.sqrt(ocean_U**2 + ocean_V**2)

    # Sample the U / V components of wind
    wind_U = fieldset.Wind_U[time, particle.depth, particle.lat, particle.lon]
    wind_V = fieldset.Wind_V[time, particle.depth, particle.lat, particle.lon]

    # Apply windage to particles that have some exposed surface above the ocean surface
    # Use a basic approach to only apply windage to particle in the ocean
    if particle.depth < 0.5*particle.plastic_diameter and ocean_speed > 1E-12:
        # Compute particle displacement
        particle_dlon += particle.wind_coefficient * (wind_U - ocean_U) * particle.dt  # noqa
        particle_dlat += particle.wind_coefficient * (wind_V - ocean_V) * particle.dt  # noqa


def SettlingVelocity(particle, fieldset, time):
    """Settling velocity kernel.

    Description
    ----------
    A kernel to calculate the settling velocity of a particle based on its
    size and density,and the surrounding seawater properties (density and
    kinematic viscosity). Based on the settling velocity paper of [1] as
    implemented in [2].

    Calculation steps:
        1. Compute the seawater dynamic viscosity from Eq. (27) in [2]
        2. Compute the kinematic viscosity from Eq. (25) in [2]
        3. Compute the dimensionless particle diameter from Eq. (4) in [2], equivalent to Eq. (6) in [1]
        4. Compute the dimensionless settling velocity from Eq. (3) in [2], equivalent to Eq. (8) in [1]
        5. Compute the settling velocity of the particle from Eq. (2) in [2], equivalent to Eq. (9) in [1]


    Parameter Requirements
    ----------
    particle :
        - diameter
        - density
        - seawater_density
    fieldset :
        - `fieldset.G` - Gravity constant. Units [m s-2].
        - `fieldset.conservative_temperature` - The conservative temperature field. Units [C].
        - `fieldset.absolute_salinity` - The absolute salinity field. Units [g/kg].


    Kernel Requirements:
    ----------
    Order of Operations:
        This kernel must run after the PolyTEOS10_bsq kernel, which sets the
        particle.seawater_density variable, relied on by this.


    References
    ----------
    [1] Dietrich (1982) - https://doi.org/10.1029/WR018i006p01615
    [2] Kooi et al. (2017) - https://doi.org/10.1021/acs.est.6b04702

    """
    # Define constants and sample fieldset variables
    g = fieldset.G  # gravitational acceleration [m s-2]
    seawater_density = particle.seawater_density  # [kg m-3]
    temperature = fieldset.conservative_temperature[time, particle.depth, particle.lat, particle.lon]
    seawater_salinity = fieldset.absolute_salinity[time, particle.depth, particle.lat, particle.lon] / 1000
    particle_diameter = particle.plastic_diameter
    particle_density = particle.plastic_density

    # Compute the kinematic viscosity of seawater
    water_dynamic_viscosity = 4.2844E-5 + (1. / ((0.156 * (temperature + 64.993) ** 2) - 91.296))  # Eq. (26) from [2]
    A = 1.541 + 1.998E-2 * temperature - 9.52E-5 * temperature ** 2  # Eq. (28) from [2]
    B = 7.974 - 7.561E-2 * temperature + 4.724E-4 * temperature ** 2  # Eq. (29) from [2]
    seawater_dynamic_viscosity = water_dynamic_viscosity * (1. + A * seawater_salinity + B * seawater_salinity ** 2)  # Eq. (27) from [2]
    seawater_kinematic_viscosity = seawater_dynamic_viscosity / seawater_density  # Eq. (25) from [2]

    # Compute the density difference of the particle
    normalised_density_difference = (particle_density - seawater_density) / seawater_density  # normalised difference in density between the plastic particle and seawater [-]

    # Compute the dimensionless particle diameter D_* using Eq. (4) from [2]
    dimensionless_diameter = (math.fabs(particle_density - seawater_density) * g * particle_diameter ** 3.) / (seawater_density * seawater_kinematic_viscosity ** 2.)  # [-]

    # Compute the dimensionless settling velocity w_*
    if dimensionless_diameter > 5E9:  # "The boundary layer around the sphere becomes fully turbulent, causing a reduction in drag and an increase in settling velocity" - [1]
        dimensionless_velocity = 265000.  # Set a maximum dimensionless settling velocity
    elif dimensionless_diameter < 0.05:  # "At values of D_* less than 0.05, (9) deviates signficantly ... from Stokes' law and (8) should be used." - [1]
        dimensionless_velocity = (dimensionless_diameter ** 2.) / 5832.  # Using Eq. (8) in [1]
    else:
        dimensionless_velocity = 10. ** (-3.76715 + (1.92944 * math.log10(dimensionless_diameter)) - (0.09815 * math.log10(dimensionless_diameter) ** 2.) - (
                                         0.00575 * math.log10(dimensionless_diameter) ** 3.) + (0.00056 * math.log10(dimensionless_diameter) ** 4.))  # Using Eq. (9) in [1]

    # Compute the settling velocity of the particle using Eq. (5) from [1] (solving for the settling velocity)
    sign_of_density_difference = math.copysign(1., normalised_density_difference)
    settling_velocity = sign_of_density_difference * (g * seawater_kinematic_viscosity * dimensionless_velocity * math.fabs(normalised_density_difference)) ** (1. / 3.)  # m s-1

    # Update the settling velocity
    particle.settling_velocity = settling_velocity

    # Update particle depth
    particle_ddepth += particle.settling_velocity * particle.dt  # noqa


def Biofouling(particle, fieldset, time):
    r"""Settling velocity due to biofouling kernel.

    Description
    ----------
    Kernel to compute the settling velocity of particles due to changes in ambient algal
    concentrations, growth and death of attached algae based on [2]. The settling velocity
    of the particle is computed as per [1], however the particle size and density is
    affected by a biofouling process. The algae attached to the particle :math:`A` has a growth
    rate described by
        :math:`\frac{dA}{dt} = C + G - M - R`

    Here, :math:`C` models fouling of the plastic through collision with algae
        :math:`C = \beta_A \cdot A_A / \theta_{pl}`,
    where :math:`\beta_A` is the encounter rate, :math:`A_A` is the ambient algal concentration, and
    :math:`\theta_{pl}` is the surface area of plastic particle.

    :math:`G` models the growth of the algae attached to the surface of the particle
        :math:`G = \mu_A \cdot A`,
    where :math:`\mu_A` is the algal growth rate.

    :math:`M` models the grazing mortality of the algae attached to the surface of the particle
        :math:`M = m_A \cdot A`,
    where :math:`m_A` is the algal mortality rate.

    :math:`R` models the respiration of the algae attached to the surface of the particle
        :math:`R = Q_{10}^{(T-20)/10}R_{20}A`,
    where :math:`Q_{10}` is the temperature coefficient, which indicates how much the respiration
    increases when the temperature increases by 10C. :math:`T` is the temperature, and :math:`R_{20}`
    is the respiration rate.

    Calculation steps:
        1. Compute the seawater dynamic viscosity from Eq. (27) in [2]
        2. Compute the kinematic viscosity from Eq. (25) in [2]
        3. Compute the dimensionless particle diameter from Eq. (4) in [2], equivalent to Eq. (6) in [1]
        4. Compute the dimensionless settling velocity from Eq. (3) in [2], equivalent to Eq. (8) in [1]
        5. Compute the settling velocity of the particle from Eq. (2) in [2], equivalent to Eq. (9) in [1]


    Parameter Requirements
    ----------
    particle :
        - diameter
        - density
        - seawater_density
    fieldset :
        - `fieldset.G` - Gravity constant. Units [m s-2].
        - `fieldset.conservative_temperature` - The conservative temperature field. Units [C].
        - `fieldset.absolute_salinity` - The absolute salinity field. Units [g/kg].
        - `fieldset.algae_cell_volume` - The volume of 1 algal cell [m-3]
        - `fieldset.biofilm_density` - The density of the biofilm [kg m-3]


    Kernel Requirements
    ----------
    Order of Operations:
        This kernel must run after the PolyTEOS10_bsq kernel, which sets the particle.seawater_density variable, relied on by this.

    References
    ----------
    [1] Dietrich (1982) - https://doi.org/10.1029/WR018i006p01615
    [2] Kooi et al. (2017) - https://doi.org/10.1021/acs.est.6b04702
    [3] Menden-Deuer and Lessard (2000) - https://doi.org/10.4319/lo.2000.45.3.0569
    [4] Bernard and Remond (2012) - https://doi.org/10.1016/j.biortech.2012.07.022
    """
    # seawater_density = particle.seawater_density  # [kg m-3]
    temperature = fieldset.conservative_temperature[time, particle.depth, particle.lat, particle.lon]
    seawater_salinity = fieldset.absolute_salinity[time, particle.depth, particle.lat, particle.lon] / 1000.
    particle_radius = 0.5 * particle.plastic_diameter
    # particle_density = particle.plastic_density
    initial_settling_velocity = particle.settling_velocity  # settling velocity [m s-1]

    # Compute the seawater dynamic viscosity and kinematic viscosity
    water_dynamic_viscosity = 4.2844E-5 + (1. / ((0.156 * (temperature + 64.993) ** 2) - 91.296))  # Eq. (26) from [2]
    A = 1.541 + 1.998E-2 * temperature - 9.52E-5 * temperature ** 2  # Eq. (28) from [2]
    B = 7.974 - 7.561E-2 * temperature + 4.724E-4 * temperature ** 2  # Eq. (29) from [2]
    seawater_dynamic_viscosity = water_dynamic_viscosity * (1. + A * seawater_salinity + B * seawater_salinity ** 2)  # Eq. (27) from [2]
    seawater_kinematic_viscosity = seawater_dynamic_viscosity / particle.seawater_density  # Eq. (25) from [2]

    # Compute the algal growth component
    # Sample fields
    mol_concentration_diatoms = fieldset.bio_diatom[time, particle.depth, particle.lat, particle.lon]  # Mole concentration of Diatoms expressed as carbon in sea water
    mol_concentration_nanophytoplankton = fieldset.bio_nanophy[time, particle.depth, particle.lat, particle.lon]  # Mole concentration of Nanophytoplankton expressed as carbon in seawater
    total_primary_production_of_phyto = fieldset.pp_phyto[time, particle.depth, particle.lat, particle.lon]  # mg C /m3/day #pp_phyto_
    median_mg_carbon_per_cell = 2726e-9  # Median mg of Carbon per cell selected from [3]. From [2] pg7966 - "The conversion from carbon to algae cells is highly variable, ranging between 35339 to 47.76 pg per carbon cell [3]. We choose the median value, 2726 x 10^-9 mg per carbon cell."
    # carbon_molecular_weight = fieldset.carbon_molecular_weight # grams C per mol of C #Wt_C

    # Compute concentration numbers
    number_concentration_diatoms = mol_concentration_diatoms * (fieldset.carbon_molecular_weight / median_mg_carbon_per_cell)  # conversion from [mmol C m-3] to [mg C m-3] to [no. m-3]
    number_concentration_diatoms = max(number_concentration_diatoms, 0.)  # Ensure non-negative
    number_concentration_nanophytoplankton = mol_concentration_nanophytoplankton * (fieldset.carbon_molecular_weight / median_mg_carbon_per_cell)
    number_concentration_nanophytoplankton = max(number_concentration_nanophytoplankton, 0.)  # Ensure non-negative
    number_concentration_total = number_concentration_diatoms + number_concentration_nanophytoplankton  # conversion from [mmol C m-3] to [mg C m-3] to [no. m-3]

    # Compute primary production
    primary_production_per_cell = total_primary_production_of_phyto / number_concentration_total  # primary productivity per cell, in mg C / cell / day
    primary_production_numcell_per_cell = primary_production_per_cell / median_mg_carbon_per_cell  # primary productivity in terms of amount of cells per cell, in cells / cell / day
    primary_production_numcell_per_cell = max(primary_production_numcell_per_cell, 0.)  # Ensure non-negative

    # Compute growth rates
    max_growth_rate = 1.85  # Maximum growth rate (per day), defined in Table S1 of [2], based on the nominal value estimated in Table 4 of [4]
    mu_a = min(primary_production_numcell_per_cell, max_growth_rate)/86400.  # Set growth rate per second

    # Compute the algal growth of the algae already on the particle
    algae_growth = mu_a * particle.algae_amount  # productivity in amount of cells/m2/s

    # Compute the radius, surface area, volume and thickness of the particle including potential biofilm
    particle_volume = (4. / 3.) * math.pi * particle_radius ** 3.  # volume of plastic [m3]
    particle_surface_area = 4. * math.pi * particle_radius ** 2.  # surface area of plastic particle [m2]
    algal_cell_radius = ((3. / 4.) * (fieldset.algae_cell_volume / math.pi)) ** (1. / 3.)  # radius of an algal cell [m]
    biofilm_volume = fieldset.algae_cell_volume * particle.algae_amount * particle_surface_area  # volume of living biofilm [m3]

    total_volume = biofilm_volume + particle_volume  # volume of total (biofilm + plastic) [m3]
    total_radius = ((total_volume * (3. / (4. * math.pi))) ** (1. / 3.))  # total radius [m]

    # Compute diffusivities
    total_density = (particle_volume * particle.plastic_density + biofilm_volume * fieldset.biofilm_density) / total_volume  # total density [kg m-3]
    plastic_diffusivity = fieldset.K * (temperature + 273.16) / (6. * math.pi * seawater_dynamic_viscosity * total_radius)  # diffusivity of plastic particle [m2 s-1]
    algae_diffusivity = fieldset.K * (temperature + 273.16) / (6. * math.pi * seawater_dynamic_viscosity * algal_cell_radius)  # diffusivity of algal cells [m2 s-1]

    # Compute the encounter rates
    beta_abrown = 4. * math.pi * (plastic_diffusivity + algae_diffusivity) * (total_radius + algal_cell_radius)  # Brownian motion [m3 s-1]
    beta_ashear = 1.3 * fieldset.Gamma * ((total_radius + algal_cell_radius) ** 3.)  # advective shear [m3 s-1]
    beta_aset = (1. / 2.) * math.pi * total_radius ** 2. * math.fabs(initial_settling_velocity)  # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset  # collision rate [m3 s-1]

    # Compute the algal growth via collision
    a_collision = fieldset.collision_probability * (beta_a * number_concentration_diatoms) / particle_surface_area  # [no. m-2 s-1] collisions with diatoms (Eq. 11 in [2])

    # Compute the algal decay due to respiration
    a_respiration = fieldset.algae_respiration_f * (fieldset.Q10 ** ((temperature - 20.) / 10.)) * fieldset.R20 * particle.algae_amount  # [no. m-2 s-1] respiration

    # Compute the algal decay due to grazing
    a_grazing = fieldset.algae_mortality_rate * particle.algae_amount

    # Compute the final algal amount
    algae_amount_change = (a_collision + algae_growth - a_grazing - a_respiration) * particle.dt
    if particle.algae_amount + algae_amount_change < 0.:
        particle.algae_amount = 0.
    else:
        particle.algae_amount += algae_amount_change

    # Compute the new settling velocity
    particle_diameter = 2. * (total_radius)  # equivalent spherical diameter [m], calculated from Dietrich (1982) from A = pi/4 * dn**2

    # Compute the density difference of the particle
    normalised_density_difference = (total_density - particle.seawater_density) / particle.seawater_density  # normalised difference in density between the plastic particle and seawater [-]

    # Compute the dimensionless particle diameter D_* using Eq. (4) from [2]
    dimensionless_diameter = (math.fabs(total_density - particle.seawater_density) * fieldset.G * particle_diameter ** 3.) / (particle.seawater_density * seawater_kinematic_viscosity ** 2.)  # [-]

    # Compute the dimensionless settling velocity w_*
    if dimensionless_diameter > 5E9:  # "The boundary layer around the sphere becomes fully turbulent, causing a reduction in drag and an increase in settling velocity" - [1]
        dimensionless_velocity = 265000.  # Set a maximum dimensionless settling velocity
    elif dimensionless_diameter < 0.05:  # "At values of D_* less than 0.05, (9) deviates signficantly ... from Stokes' law and (8) should be used." - [1]
        dimensionless_velocity = (dimensionless_diameter ** 2.) / 5832.  # Using Eq. (8) in [1]
    else:
        dimensionless_velocity = 10. ** (-3.76715 + (1.92944 * math.log10(dimensionless_diameter)) - (0.09815 * math.log10(dimensionless_diameter) ** 2.) - (
                                         0.00575 * math.log10(dimensionless_diameter) ** 3.) + (0.00056 * math.log10(dimensionless_diameter) ** 4.))  # Using Eq. (9) in [1]

    # Compute the settling velocity of the particle using Eq. (5) from [1] (solving for the settling velocity)
    sign_of_density_difference = math.copysign(1., normalised_density_difference)
    settling_velocity = sign_of_density_difference * (fieldset.G * seawater_kinematic_viscosity * dimensionless_velocity * math.fabs(normalised_density_difference)) ** (1. / 3.)  # m s-1

    # Update the settling velocity
    particle.settling_velocity = settling_velocity

    # Update particle depth
    particle_ddepth += particle.settling_velocity * particle.dt  # noqa


def PolyTEOS10_bsq(particle, fieldset, time):
    """A seawater density kernel.

    Description:
    ----------
    A kernel to calculate the seawater density surrounding a particle based on
    the polyTEOS10-bsq algorithm from Appendix A.2 of
    https://doi.org/10.1016/j.ocemod.2015.04.002

    Parameter Requirements
    ----------
    particle :
        - seawater_density
    fieldset :
        - `fieldset.conservative_temperature` - The conservative temperature field. Units [C].
        - `fieldset.absolute_salinity` - The absolute salinity field. Units [g/kg].

    Kernel Requirements:
    ----------
    Order of Operations:
        This kernel must be run before any kernel that requires an updated `particle.seawater_density` value.

    References
    ----------
     Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
      polynomial expressions for the density and specific volume of
      seawater using the TEOS-10 standard. Ocean Modelling.
     McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
      Accurate and computationally efficient algorithms for potential
      temperature and density of seawater.  Journal of Atmospheric and
      Oceanic Technology, 20, 730-741.

    """
    Z = - math.fabs(particle.depth)  # note: use negative depths!
    SA = fieldset.absolute_salinity[time, particle.depth, particle.lat, particle.lon]
    CT = fieldset.conservative_temperature[time, particle.depth, particle.lat, particle.lon]

    SAu = 40. * 35.16504 / 35.
    CTu = 40.
    Zu = 1.0e+04
    deltaS = 32.

    zz = -Z / Zu
    R00 = 4.6494977072e+01
    R01 = -5.2099962525e+00
    R02 = 2.2601900708e-01
    R03 = 6.4326772569e-02
    R04 = 1.5616995503e-02
    R05 = -1.7243708991e-03
    r0 = (((((R05 * zz + R04) * zz + R03) * zz + R02) * zz + R01) * zz + R00) * zz

    R000 = 8.0189615746e+02
    R100 = 8.6672408165e+02
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03
    R400 = -1.2849161071e+03
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01
    R010 = 2.6010145068e+01
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00
    R020 = -3.7074170417e+01
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01
    R130 = -3.3449108469e+01
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00
    R040 = -8.3627885467e+00
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00
    R050 = 5.4048723791e-01
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01
    R001 = 1.9681925209e+01
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01
    R301 = -3.0938076334e+01
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00
    R031 = -2.4649669534e+00
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00
    R003 = -2.3342758797e-02
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01
    ss = math.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    zz = -Z / Zu
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
    rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
    rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
    r = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0

    particle.seawater_density = r0 + r


def VerticalMixing(particle, fieldset, time):
    """A markov-0 kernel for vertical mixing.

    Description
    ----------
    A simple verticle mixing kernel that uses a markov-0 process to determine the vertical
    displacement of a particle [1]. The deterministic component is determined
    using forward-difference with a given `delta_z`.

    Parameter Requirements
    ----------
    particle:
        - settling_velocity
    fieldset:
        - `fieldset.mixing_kz` - The vertical eddy diffusivity field. Units [m2 s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        To ensure the reflecting boundary condition of the random walk component, this kernel should be performed at the very end.
        Additionally, this kernel should be performed after the rising/sinking velocity of the particle has been computed.

    References
    ----------
    [1] Ross & Sharples (2004) - https://doi.org/10.4319/lom.2004.2.289

    """
    # Sample the ocean vertical eddy diffusivity field KZ
    delta_z = 0.5  # [m], used to compute the gradient of kz using a forward-difference
    kz = fieldset.mixing_kz[time, particle.depth, particle.lat, particle.lon]
    kz_delta = fieldset.mixing_kz[time, particle.depth+delta_z, particle.lat, particle.lon]

    # Compute the gradient of kz using a forward-difference
    dkz_dz = (kz_delta - kz) / delta_z

    # Compute the deterministic component of Eq. (1)
    dz_deterministic = dkz_dz * particle.dt

    # Compute the random walk component of Eq. (1)
    dz_random = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3) * math.sqrt(2 * kz)
    # TODO - implement the reflective boundary condition

    # Compute rise velocity component of Eq. (1) - Already accounted for in other kernels
    dz_wb = 0  # particle.settling_velocity * particle.dt

    # Apply Eq. (1) - computing only the difference in depth
    ddepth = dz_deterministic + dz_random + dz_wb

    # Update particle position
    particle_ddepth += ddepth  # noqa

# Biofouling related kernels


def unbeaching(particle, fieldset, time):
    """Unbeaching kernel.

    Description
    ----------
    A simple kernel to 'unbeach' particles that have been advected onto non-ocean grid cells.
    This kernel uses a simple 'unbeaching' field, that provides (U,V) velocities with a magnitude
    of 1 m/s, normal to the grid cell face between an ocean and non-ocean grid cell.

    Parameter Requirements
    ----------
    Fieldset:
        - `fieldset.unbeach_U` - The zonal unbeaching velocity. Units [m s-1].
        - `fieldset.unbeach_V` - The meridional unbeaching velocity. Units [m s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it samples the
        unbeaching field using the updated particle position.
    """
    # Measure the velocity field at the final particle location
    (vel_u, vel_v, vel_w) = fieldset.UVW[time, particle.depth + particle_ddepth, particle.lat + particle_dlat, particle.lon + particle_dlon]  # noqa

    if math.fabs(vel_u) < 1e-14 and math.fabs(vel_v) < 1e-14:
        U_ub = fieldset.unbeach_U[time, particle.depth + particle_ddepth, particle.lat + particle_dlat, particle.lon + particle_dlon]  # noqa
        V_ub = fieldset.unbeach_V[time, particle.depth + particle_ddepth, particle.lat + particle_dlat, particle.lon + particle_dlon]  # noqa

        dlon = U_ub * particle.dt
        dlat = V_ub * particle.dt

        particle_dlon += dlon  # noqa
        particle_dlat += dlat  # noqa


def checkThroughBathymetry(particle, fieldset, time):
    """Bathymetry error kernel.

    Description
    ----------
    A simple kernel to ensure particles are not advected below the bathymetry field.

    Parameter Requirements
    ----------
    Fieldset:
        - `fieldset.z_start` - A field constant representing the minimum depth. Units [m].
        - `fieldset.bathymetry` - A 2D field containing the ocean bathymetry. Units [m].

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it samples the
        bathymetry field using the updated particle position.
    """
    bathymetry_local = fieldset.bathymetry[time, particle.depth + particle_ddepth, particle.lat + particle_dlat, particle.lon + particle_dlon]  # noqa
    potential_depth = particle.depth + particle_ddepth  # noqa
    min_depth = 0.5  # meters - maybe set this as a fieldset variable?

    if potential_depth < min_depth:
        particle_ddepth = fieldset.z_start - particle.depth  # noqa # Stick particle to surface
    elif potential_depth > bathymetry_local:
        particle_ddepth = bathymetry_local - particle.depth  # noqa # Stick particle at bottom of ocean
    elif particle.depth > 100. and potential_depth > (bathymetry_local*0.99):  # for deeper particles; since bathymetry can be quite rough (and is interpolated linearly) look at the 99% value instead
        particle_ddepth = bathymetry_local*0.99 - particle.depth  # noqa # Stick particle at the 99% point

    elif potential_depth > 3900.:  # If particle >3.9km deep, stick it there
        particle_ddepth = 3900. - particle.depth  # noqa



def periodicBC(particle, fieldset, time):
    """A periodic boundary condition kernel.

    Description
    ----------
    Kernel to keep the particle between [-180,180] longitude

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it sets the updated particle
        longitude to be within the [-180,180] longitudinal range.
    """
    if particle.lon + particle_dlon <= -180.:  # noqa
        particle_dlon += 360.  # noqa
    elif particle.lon + particle_dlon > 180.:  # noqa
        particle_dlon -= 360.  # noqa


def checkErrorThroughSurface(particle, fieldset, time):
    """Surface error kernel.

    Description
    ----------
    Kernel to place a particle back on the surface if it goes through the surface.

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it is an error kernel.
    """
    if particle.state == StatusCode.ErrorThroughSurface:
        particle_ddepth = 0. # Set the summed displacement to 0.
        particle_ddepth += fieldset.z_start - particle.depth # Set ddepth so that final depth is the ocean surface
        particle.state = StatusCode.Success


def deleteParticle(particle, fieldset, time):
    """General error kernel.

    Description
    ----------
    Kernel to delete a particle if it throws an error other than `ErrorThroughSurface`.

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it is an error kernel.
    """
    if particle.state >= 50 and particle.state != StatusCode.ErrorThroughSurface:
        particle.delete()


def delete_particle_interp(particle, fieldset, time):
    """DEPRECATED: Interpolation error kernel.

    Description
    ----------
    Kernel to delete a particle if it throws an interpolation error.

    Kernel Requirements
    ----------
    Order of Operations:
        This kernel should be performed after all other movement kernels, as it is an error kernel.
    """
    if fieldset.verbose_delete == 1:
        print('particle is deleted due to an interpolation error at lon = ' + str(particle.lon) + ', lat =' + str(
            particle.lat) + ', depth =' + str(particle.depth))

    particle.delete()
