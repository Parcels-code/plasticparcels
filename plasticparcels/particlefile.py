from parcels import ParticleFile as pcls_particlefile


def ParticleFile(name, outputdt, settings, **kwargs):
    """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet."""
    pfile = pcls_particlefile(name, outputdt=outputdt, **kwargs)
    pfile.metadata.update({'plasticparcels_settings': str(settings)})
    return pfile
