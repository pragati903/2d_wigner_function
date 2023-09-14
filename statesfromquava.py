import jax.numpy as jnp
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



def get_state(statename):
    filename = f'STATES/{statename}'
    jaxstate = jnp.array(np.load(f'{filename}.npy'))
    jaxstatedm = jnp.outer(jaxstate,jnp.conjugate(jnp.transpose(jaxstate)))
    jaxstatedm_bipartite = jnp.reshape(jaxstatedm,[40,40,40,40]).transpose()
    return jaxstatedm_bipartite



if __name__ == '__main__':

    filename = 'STATES/tesseract_mx'
    jaxstate = jnp.array(np.load(f'{filename}.npy'))
    jaxstatedm = jnp.outer(jaxstate,jnp.conjugate(jnp.transpose(jaxstate)))
    jaxstatedm_bipartite = jnp.reshape(jaxstatedm,[40,40,40,40]).transpose()
    jaxstate1 = jnp.trace(jaxstatedm_bipartite, axis1=0, axis2 =2 )
    jaxstate2 = jnp.trace(jaxstatedm_bipartite, axis1=1, axis2 =3 )


    jaxstatefull = (jnp.kron(jaxstate, jnp.array([[1],[0]])))
    jaxstatefulldm = jnp.outer(jaxstatefull,jnp.conjugate(jnp.transpose(jaxstatefull)))
    jaxstatefulldm_bipartite = jnp.reshape(jaxstatefulldm,[1600, 2, 1600, 2]).transpose()
    jaxstatefull12 = jnp.trace(jaxstatefulldm_bipartite, axis1=0, axis2 =2 )
    jaxstatefull12 = jnp.reshape(jaxstatefull12,[40,40,40,40]).transpose()
    jaxstatefull1 = jnp.trace(jaxstatefull12, axis1=0, axis2 =2 )
    jaxstatefull2 = jnp.trace(jaxstatefull12, axis1=1, axis2 =3 )
    print(jaxstatefull2- jaxstate2);

    xvec = np.linspace(-10,10,100)
    w = qt.wigner(qt.Qobj(np.array(jaxstatefull2)), xvec,xvec)
    fig, ax = plt.subplots()
    ax.contourf(xvec,xvec,w)
    plt.show()

    op = jnp.array(qt.tensor(qt.destroy(40),
                            qt.identity(40),
                            qt.identity(2)).full())
    print(op.shape)
