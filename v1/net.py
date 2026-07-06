#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch
from mininet.link import TCLink  # Import the Traffic Control Link class
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def launchNetwork():
    # Initialize network
    net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8', link=TCLink)

    info('*** Adding controller\n')
    net.addController(name='c0', controller=Controller)

    info('*** Adding switches\n')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)

    info('*** Adding hosts\n')
    h1 = net.addHost('h1', ip='10.0.0.1')  # Target Server
    h2 = net.addHost('h2', ip='10.0.0.2')  # Legitimate User A
    h3 = net.addHost('h3', ip='10.0.0.3')  # Legitimate User B
    h4 = net.addHost('h4', ip='10.0.0.4')  # Attacker A
    h5 = net.addHost('h5', ip='10.0.0.5')  # Attacker B

    info('*** Creating links with bandwidth limits\n')
    # Limit the server's bottleneck link to 10 Mbps
    net.addLink(h1, s1, bw=10) 
    
    # Other users get standard links
    for host in [h2, h3, h4, h5]:
        net.addLink(host, s1, bw=100)

    info('*** Starting network\n')
    net.start()
    
    info('*** Network is ready. Dropping into Mininet CLI...\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    launchNetwork()
