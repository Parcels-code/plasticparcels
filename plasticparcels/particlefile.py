from parcels import ParticleFile as pcls_particlefile
from parcels import read_particlefile as pcls_read_particlefile


def ParticleFile(name, outputdt, settings, **kwargs):
    """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet."""
    pfile = pcls_particlefile(name, outputdt=outputdt, **kwargs)
    pfile.metadata.update({'plasticparcels_settings': str(settings)})
    return pfile

def read_particlefile(name, **kwargs):
    """Wrapper method to read a :class:`parcels.particlefile.ParticleFile` parquet file."""
    return pcls_read_particlefile(name, **kwargs)