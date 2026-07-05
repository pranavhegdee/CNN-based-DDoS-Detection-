import json
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

class RyuApiServer(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(RyuApiServer, self).__init__(req, link, data, **config)
        self.ryu_app = data['ryu_app']

    @route('mitigate', '/mitigate', methods=['POST'])
    def inject_block_rule(self, req, **kwargs):
        try:
            body = json.loads(req.body.decode('utf-8'))
            attacker_ip = body.get('ip')
            
            if attacker_ip:
                self.ryu_app.block_ip_at_hw(attacker_ip)
                return Response(status=200, body=json.dumps({"status": "VACL rule deployed", "blocked": attacker_ip}))
            return Response(status=400, body=json.dumps({"error": "Missing IP parameter"}))
        except Exception as e:
            return Response(status=500, body=str(e))

class AutomatedAclController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(AutomatedAclController, self).__init__(*args, **kwargs)
        self.switches = {}
        wsgi = kwargs['wsgi']
        wsgi.register(RyuApiServer, {'ryu_app': self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.switches[datapath.id] = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Default Flow Rule (Priority 0): Forward unmatched packets normally
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=0, match=match, instructions=inst)
        datapath.send_msg(mod)
        print(f"[+] Switch {datapath.id} initialized with standard OpenFlow 1.3 paths.")

    def block_ip_at_hw(self, attacker_ip):
        """Pushes a high-priority hardware DROP rule via OpenFlow."""
        for datapath in self.switches.values():
            parser = datapath.ofproto_parser
            
            # Match IPv4 packets where Source IP equals the attacker
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=attacker_ip)
            
            # Empty actions list = DROPPING PACKET (Wire-Speed VACL)
            actions = []
            
            # Priority 10000 overrides default normal forwarding routing
            inst = [parser.OFPInstructionActions(ofproto_v1_3.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(datapath=datapath, priority=10000, match=match, instructions=inst)
            datapath.send_msg(mod)
            print(f"[🛡️ VACL ACTIVE] OpenFlow pushed: Dropping {attacker_ip} at switch level.")
