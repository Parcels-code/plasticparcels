# Physics kernels

The [`parcels`](https://oceanparcels.org/) Lagrangian framework is a tool for advecting virtual particles that are assumed to be spherical in shape. It works by numerically integrating the velocity fields from a hydrodynamic model while including any additional \textit{behaviour} of the particle. Mathematically, particle trajectories are computed by solving the following equation:

```{math}
\mathbf{x}(t) = \mathbf{x}(0) + \int_{0}^{t} \mathbf{v}(\mathbf{x}(s), s) + \mathbf{B}(\mathbf{x}(s),s) \text{d}s,
```

where {math}`\mathbf{x}(t)` describes the particle position at time {math}`t`, {math}`\mathbf{v} = (u,v,w)` is the hydrodynamic model velocity field, and {math}`\mathbf{B}(\mathbf{x}(t),t)` describes any displacements to the particle position caused by additional behaviour the particle exhibits or experiences. When performing a plastic dispersal simulation with `plasticparcels`, users have the explicit option of choosing which additional behaviour to include. Examples of these additional behaviours are described below.

Numerically, we solve the above equation using a time-stepping approach, where we compute the displacements in the particle position as

```{math}
\frac{\text{d}\mathbf{x}(t)}{\text{d}t} = \mathbf{v}(\mathbf{x}(t), t) + \mathbf{B}(\mathbf{x}(t), t),
```

and updating the particle position at each timestep. For simplicity, by default we use the fourth-order Runge-Kutta scheme of [`parcels`](https://oceanparcels.org/) to solve the advection of the particle from the hydrodynamic model velocity field {math}`\mathbf{v}`, and an Euler-forward scheme for all other additional behaviours realised in {math}`\mathbf{B}`.


### Stokes drift

An important process that affects plastic particle dispersal in the upper ocean is the Stokes drift, whereby a particle subjected to a surface wave will experience a net displacement in the direction of wave propagation. We include a kernel to parameterise the effect of Stokes drift on a particle, based on the Phillips spectrum approximation developed in [@Breivik2016](http://dx.doi.org/10.1016/j.ocemod.2016.01.005). Specifically, we model this additional behaviour as {math}`\mathbf{B}_{\text{Stokes}}`, where the change in the particle position is described by

```{math}
\mathbf{B}_{\text{Stokes}} := \mathbf{v}_{\text{Stokes}}(\mathbf{x}(t), t) =\mathbf{v}_{\text{Stokes}}(\mathbf{x}_{z=0}(t),t)\bigg(e^{2k_p z} - \beta \sqrt{-2\pi k_p z}\text{ erfc}(-2k_p z) \bigg).
```

Here, {math}`z` is the depth of the particle, {math}`\mathbf{v}_{\text{Stokes}}(\mathbf{x}_{z=0}(t),t)` is the surface Stokes drift velocity, {math}`\beta=1` (as we assume a Phillips spectrum), and erfc is the complementary error function. The peak wave number {math}`k_p` is computed as {math}`k_p = \omega_{p}^2/9.81`, where {math}`\omega_p` is the peak wave frequency {math}`\omega_p = 2 \pi / T_p`, using the peak wave period {math}`T_p = T_p(\mathbf{x}_{z=0}(t),t)`.

Our particular implementation of the Stokes drift kernel requires a surface Stokes velocity field {math}`\mathbf{v}_{\text{Stokes}}(\mathbf{x}_{z=0}(t),t)`, as well as a peak wave period field {math}`T_p(\mathbf{x}_{z=0}(t),t)`. Earlier versions of this kernel have been used in [@Onink2022](https://pubs.acs.org/doi/full/10.1021/acs.est.2c03363).


### Wind-induced drift / Leeway
Plastic particles at the ocean surface that are not completely submersed will experience a force from the relative wind due to a wind drag, leading to a wind-induced drift. This wind-induced drift of the particle is called leeway [@Allen1999](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA366414.xhtml), which can be decomposed into a downwind component (in the direction of the wind), and a crosswind component (which is typically non-zero for asymmetric objects). As we assume that each plastic particle is spherical, we can ignore the crosswind component of leeway, and only consider the downwind component of leeway. The downwind component follows an almost linear relationship with the relative 10m wind speed [@Allen2005](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA435435.xhtml), so we model the leeway as


```{math}
\mathbf{B}_{\text{Wind}} := c \cdot \big(\mathbf{v}_{\text{Wind}}(\mathbf{x}(t),t) - \mathbf{v}(\mathbf{x}(t),t)\big),
```


where {math}`\mathbf{v}_{\text{Wind}}` is the wind velocity 10m above sea level, and {math}`c` is the leeway rate (a percentage of wind speed, which we refer to in the code as the windage coefficient). Ignoring all additional behaviour of the particle, then {math}`\mathbf{v}_{\text{Wind}} - \mathbf{v}` is the relative wind acting on the particle.

A version of this kernel has been used in @[Manral2024](https://open-research-europe.ec.europa.eu/articles/4-41).

### Biofouling
Plastic particles in the ocean can be a hotbed for the accumulation and growth of organisms, known as biofouling. The formation of a biofilm on the surface of a plastic particle can result in a density change, affecting the buoyancy of the particle. An initially buoyant particle may become negatively buoyant, and sink or settle, depending on the surrounding seawater density.

We model the biofouling of a plastic particle following the approach of [@Kooi2017](http://dx.doi.org/10.1021/acs.est.6b04702), where the settling velocity of a particle is computed from the relative density difference of the plastic particle and the surrounding seawater. Here, we assume the biofilm growth (and decay) is primarily microbial algae, and is distributed homogeneously over the particle surface. The density of the biofouled plastic particle depends on the radius and density of the particle, and the thickness and density of the algal biofilm. The primary component of the biofouling kernel is modelling the change in the number of attached algae (denoted by {math}`A`) on the surface of the plastic particle. As in [@Kooi2017](http://dx.doi.org/10.1021/acs.est.6b04702), we model the attached algal growth as


```{math}
\frac{\text{d}A}{\text{d}t} := \underbrace{\frac{A_A \beta_A}{\theta_\text{Plastic}}}_{\text{Collisions}} + \overbrace{\mu_A A}^{\text{Algal growth}} - \underbrace{m_A A}_{\text{Mortality}} - \overbrace{Q_{10}^{(T-20)/10}R_{20}A}^{\text{Respiration}}.
```


The first term models growth of algae due to collisions of the particle with algae in the surrounding seawater, where {math}`A_A` is the ambient algal amount, {math}`\beta_A` is the encounter kernel rate, {math}`\theta_{\text{Plastic}}` is the surface area of the plastic particle. The second term models the growth of the biofilm, where the growth term {math}`\mu_A` is computed from the total productivity provided by model output. The third and fourth terms model the (grazing) mortality and respiration of the biofilm respectively. As in [@Kooi2017](http://dx.doi.org/10.1021/acs.est.6b04702), we use constant mortality {math}`m_A` and respiration {math}`R_{20}` rates, with a temperature dependent term {math}`\big(Q_{10}^{(T-20)/10}\big)` included in the respiration component (see [@Kooi2017](http://dx.doi.org/10.1021/acs.est.6b04702) for more details).

As described above, the modelled attached algal growth drives a change in the settling velocity of the biofouled particle, {math}`\mathbf{v}_{\text{Biofouling}}`. Hence, we model the additional behaviour of the particle due to biofouling as

```{math}
\mathbf{B}_{\text{Biofouling}} := \mathbf{v}_{\text{Biofouling}}.
```

This kernel has been used in various forms in [@Lobelle2021](http://dx.doi.org/10.1029/2020JC017098), [@Fischer2022](http://dx.doi.org/10.5194/bg-19-2211-2022), and [@Kaandorp2023](http://dx.doi.org/10.1038/s41561-023-01216-0).

### Vertical mixing
An important process that is unresolved in even high-resolution ocean models is wind-driven turbulent mixing, which occurs at scales far smaller than a typical model ocean grid cell. In the vertical direction, this turbulent mixing can distribute even positively buoyant plastic particles throughout the mixed layer. To model this process, we take the approach of [@Onink2022](http://dx.doi.org/10.5194/gmd-15-1995-2022), by employing a Markov-0 styled stochastic parameterisation.

Denote by {math}`K_z = K_z(\mathbf{x}(t))` the vertical diffusion coefficient profile based on a {math}`K`-profile parameterisation (KPP) model [@Large1994](http://dx.doi.org/10.1029/94RG01872). Then the displacement of a particle (in the vertical direction) with a settling velocity {math}`w` can be modelled as an SDE [@Grawe2012](http://dx.doi.org/10.1007/s10236-012-0523-y),

```{math}
\mathbf{B}_{\text{Vertical Mixing}} := \bigg(w + \frac{\partial K_z}{\partial z}\bigg)\text{d}t + \sqrt{2 K_z}\text{d}W(t),
```

where {math}`\text{d}W(t)` is a Wiener noise increment with zero mean and a variance of {math}`\text{d}t`. In our case, the displacement due to the settling velocity of a particle is already accounted for in the biofouling kernel, hence we only model the stochastic term (by setting {math}`w=0`). To numerically solve this equation, we use the stochastic generalisation of the Euler-forward scheme, called the Euler-Maruyama scheme [@Maruyama1955](http://dx.doi.org/10.1007/BF02846028).

A version of this kernel has been used in [@Onink2022](http://dx.doi.org/10.5194/gmd-15-1995-2022).

### Fecal Pellets Egestion
Plastic particles are susceptible to be ingested and excreted by small organisms belonging to the zooplankton category.
After the egestion, they are incorporated into fecal pellets that differ in density from both virgin plastic and pure organic material. The presence of organic matter surrounding the plastic particle leads to a change in its overall density, thereby affecting its buoyancy. A particle that was initially buoyant may become negatively buoyant and sink or settle, depending on the density of the surrounding seawater. The reduction in the volume of organic matter leads to changes in the particle's density and, consequently, in its settling velocity, until the organic matter is completely degraded and the plastic particle returns to its initial state. The modeling of the dynamics of egested fecal pellets (+ plastic particles) has been developed following approach of [@Omand2020](https://doi.org/10.1038/s41598-020-60424-5).

In this framework, the settling velocity of a particle is computed based on the difference between the particle's density and that of the surrounding seawater. We assume that the organic matter enveloping the plastic particle is remineralized, as time passes by, according to the equation describing the variation of its equivalent radius: {math}`a`:

```{math}
\frac{\text{d}a}{\text{d}t} := - \text{C} \text{r} \, a^{\text{n}-2}.
```

Where  \text{C}, \text{r} and \text{n} are the remineralization constant, the remineralization rate and a constant related to the type of remineralization.


### Sea-ice capture
A sea-ice capture kernel is currently under development and will be released soon.
