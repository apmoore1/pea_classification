from pathlib import Path
from typing import List

from pea_classification.dataset_util import _get_from_cache

US2UK = {
 'favorable': 'favourable',
 'favorably': 'favourably',
 'favored': 'favoured',
 'favoring': 'favouring',
 'favorite': 'favourite',
 'favorites': 'favourites',
 'stabilizations': 'stabilisations',
 'disfavored': 'disfavoured',
 'disfavoring': 'disfavouring',
 'disfavors': 'disfavours',
 'jeopardize': 'jeopardise',
 'jeopardized': 'jeopardised',
 'mischaracterization': 'mischaracterisation',
 'monopolization': 'monopolisation',
 'monopolize': 'monopolise',
 'monopolized': 'monopolised',
 'monopolizes': 'monopolises',
 'monopolizing': 'monopolising',
 'unauthorized': 'unauthorised',
 'undercapitalized': 'undercapitalised',
 'unfavorability': 'unfavourability',
 'unfavorable': 'unfavourable',
 'unfavorably': 'unfavourably',
 'utilizing': 'utilising',
 'accessorize': 'accessorise',
 'accessorized': 'accessorised',
 'accessorizes': 'accessorises',
 'accessorizing': 'accessorising',
 'acclimatization': 'acclimatisation',
 'acclimatize': 'acclimatise',
 'acclimatized': 'acclimatised',
 'acclimatizes': 'acclimatises',
 'acclimatizing': 'acclimatising',
 'accouterments': 'accoutrements',
 'aerogram': 'aerogramme',
 'aerograms': 'aerogrammes',
 'aggrandizement': 'aggrandisement',
 'aging': 'ageing',
 'agonize': 'agonise',
 'agonized': 'agonised',
 'agonizes': 'agonises',
 'agonizing': 'agonising',
 'agonizingly': 'agonisingly',
 'airplane': 'aeroplane',
 'airplanes ': 'aeroplanes ',
 'almanac': 'almanack',
 'almanacs': 'almanacks',
 'aluminum': 'aluminium',
 'amortizable': 'amortisable',
 'amortization': 'amortisation',
 'amortizations': 'amortisations',
 'amortize': 'amortise',
 'amortized': 'amortised',
 'amortizes': 'amortises',
 'amortizing': 'amortising',
 'amphitheater': 'amphitheatre',
 'amphitheaters': 'amphitheatres',
 'analog': 'analogue',
 'analogs': 'analogues',
 'analyze': 'analyse',
 'analyzed': 'analysed',
 'analyzes': 'analyses',
 'analyzing': 'analysing',
 'anemia': 'anaemia',
 'anemic': 'anaemic',
 'anesthesia': 'anaesthesia',
 'anesthetic': 'anaesthetic',
 'anesthetics': 'anaesthetics',
 'anesthetist': 'anaesthetist',
 'anesthetists': 'anaesthetists',
 'anesthetize': 'anaesthetize',
 'anesthetized': 'anaesthetized',
 'anesthetizes': 'anaesthetizes',
 'anesthetizing': 'anaesthetizing',
 'anglicize': 'anglicise',
 'anglicized': 'anglicised',
 'anglicizes': 'anglicises',
 'anglicizing': 'anglicising',
 'annualized': 'annualised',
 'antagonize': 'antagonise',
 'antagonized': 'antagonised',
 'antagonizes': 'antagonises',
 'antagonizing': 'antagonising',
 'apologize': 'apologise',
 'apologized': 'apologised',
 'apologizes': 'apologises',
 'apologizing': 'apologising',
 'appall': 'appal',
 'appalls': 'appals',
 'appetizer': 'appetiser',
 'appetizers': 'appetisers',
 'appetizing': 'appetising',
 'appetizingly': 'appetisingly',
 'arbor': 'arbour',
 'arbors': 'arbours',
 'archeological': 'archaeological',
 'archeologically': 'archaeologically',
 'archeologist': 'archaeologist',
 'archeologists': 'archaeologists',
 'archeology': 'archaeology',
 'ardor': 'ardour',
 'armor': 'armour',
 'armored': 'armoured',
 'armorer': 'armourer',
 'armorers': 'armourers',
 'armories': 'armouries',
 'armory': 'armoury',
 'artifact': 'artefact',
 'artifacts': 'artefacts',
 'authorize': 'authorise',
 'authorized': 'authorised',
 'authorizes': 'authorises',
 'authorizing': 'authorising',
 'ax': 'axe',
 'backpedaled': 'backpedalled',
 'backpedaling': 'backpedalling',
 'balk': 'baulk',
 'balked': 'baulked',
 'balking': 'baulking',
 'balks': 'baulks',
 'banister': 'bannister',
 'banisters': 'bannisters',
 'baptize': 'baptise',
 'baptized': 'baptised',
 'baptizes': 'baptises',
 'baptizing': 'baptising',
 'bastardize': 'bastardise',
 'bastardized': 'bastardised',
 'bastardizes': 'bastardises',
 'bastardizing': 'bastardising',
 'battleax': 'battleaxe',
 'bedeviled': 'bedevilled',
 'bedeviling': 'bedevilling',
 'behavior': 'behaviour',
 'behavioral': 'behavioural',
 'behaviorism': 'behaviourism',
 'behaviorist': 'behaviourist',
 'behaviorists': 'behaviourists',
 'behaviors': 'behaviours',
 'behoove': 'behove',
 'behooved': 'behoved',
 'behooves': 'behoves',
 'bejeweled': 'bejewelled',
 'belabor': 'belabour',
 'belabored': 'belaboured',
 'belaboring': 'belabouring',
 'belabors': 'belabours',
 'beveled': 'bevelled',
 'bevies': 'bevvies',
 'bevy': 'bevvy',
 'biased': 'biassed',
 'biasing': 'biassing',
 'binging': 'bingeing',
 'bougainvillea': 'bougainvillaea',
 'bougainvilleas': 'bougainvillaeas',
 'bowdlerize': 'bowdlerise',
 'bowdlerized': 'bowdlerised',
 'bowdlerizes': 'bowdlerises',
 'bowdlerizing': 'bowdlerising',
 'breathalyze': 'breathalyse',
 'breathalyzed': 'breathalysed',
 'breathalyzer': 'breathalyser',
 'breathalyzers': 'breathalysers',
 'breathalyzes': 'breathalyses',
 'breathalyzing': 'breathalysing',
 'brutalize': 'brutalise',
 'brutalized': 'brutalised',
 'brutalizes': 'brutalises',
 'brutalizing': 'brutalising',
 'busses': 'buses',
 'bussing': 'busing',
 'caliber': 'calibre',
 'calibers': 'calibres',
 'caliper': 'calliper',
 'calipers': 'callipers',
 'calisthenics': 'callisthenics',
 'canalize': 'canalise',
 'canalized': 'canalised',
 'canalizes': 'canalises',
 'canalizing': 'canalising',
 'cancelation': 'cancellation',
 'cancelations': 'cancellations',
 'canceled': 'cancelled',
 'canceling': 'cancelling',
 'candor': 'candour',
 'cannibalize': 'cannibalise',
 'cannibalized': 'cannibalised',
 'cannibalizes': 'cannibalises',
 'cannibalizing': 'cannibalising',
 'canonize': 'canonise',
 'canonized': 'canonised',
 'canonizes': 'canonises',
 'canonizing': 'canonising',
 'capitalize': 'capitalise',
 'capitalized': 'capitalised',
 'capitalizes': 'capitalises',
 'capitalizing': 'capitalising',
 'caramelize': 'caramelise',
 'caramelized': 'caramelised',
 'caramelizes': 'caramelises',
 'caramelizing': 'caramelising',
 'carbonize': 'carbonise',
 'carbonized': 'carbonised',
 'carbonizes': 'carbonises',
 'carbonizing': 'carbonising',
 'caroled': 'carolled',
 'caroling': 'carolling',
 'catalog': 'catalogue',
 'cataloged': 'catalogued',
 'cataloging': 'cataloguing',
 'catalogs': 'catalogues',
 'catalyze': 'catalyse',
 'catalyzed': 'catalysed',
 'catalyzes': 'catalyses',
 'catalyzing': 'catalysing',
 'categorize': 'categorise',
 'categorized': 'categorised',
 'categorizes': 'categorises',
 'categorizing': 'categorising',
 'cauterize': 'cauterise',
 'cauterized': 'cauterised',
 'cauterizes': 'cauterises',
 'cauterizing': 'cauterising',
 'caviled': 'cavilled',
 'caviling': 'cavilling',
 'center': 'centre',
 'centered': 'centred',
 'centerfold': 'centrefold',
 'centerfolds': 'centrefolds',
 'centerpiece': 'centrepiece',
 'centerpieces': 'centrepieces',
 'centers': 'centres',
 'centigram': 'centigramme',
 'centigrams': 'centigrammes',
 'centiliter': 'centilitre',
 'centiliters': 'centilitres',
 'centimeter': 'centimetre',
 'centimeters': 'centimetres',
 'centralize': 'centralise',
 'centralized': 'centralised',
 'centralizes': 'centralises',
 'centralizing': 'centralising',
 'cesarean': 'caesarean',
 'cesareans': 'caesareans',
 'channeled': 'channelled',
 'channeling': 'channelling',
 'characterize': 'characterise',
 'characterized': 'characterised',
 'characterizes': 'characterises',
 'characterizing': 'characterising',
 'check': 'cheque',
 'checkbook': 'chequebook',
 'checkbooks': 'chequebooks',
 'checkered': 'chequered',
 'checks': 'cheques',
 'chili': 'chilli',
 'chimera': 'chimaera',
 'chimeras': 'chimaeras',
 'chiseled': 'chiselled',
 'chiseling': 'chiselling',
 'cipher': 'cypher',
 'ciphers': 'cyphers',
 'circularize': 'circularise',
 'circularized': 'circularised',
 'circularizes': 'circularises',
 'circularizing': 'circularising',
 'civilize': 'civilise',
 'civilized': 'civilised',
 'civilizes': 'civilises',
 'civilizing': 'civilising',
 'clamor': 'clamour',
 'clamored': 'clamoured',
 'clamoring': 'clamouring',
 'clamors': 'clamours',
 'clangor': 'clangour',
 'clarinetist': 'clarinettist',
 'clarinetists': 'clarinettists',
 'collectivize': 'collectivise',
 'collectivized': 'collectivised',
 'collectivizes': 'collectivises',
 'collectivizing': 'collectivising',
 'colonization': 'colonisation',
 'colonize': 'colonise',
 'colonized': 'colonised',
 'colonizer': 'coloniser',
 'colonizers': 'colonisers',
 'colonizes': 'colonises',
 'colonizing': 'colonising',
 'color': 'colour',
 'colorant': 'colourant',
 'colorants': 'colourants',
 'colored': 'coloured',
 'coloreds': 'coloureds',
 'colorful': 'colourful',
 'colorfully': 'colourfully',
 'coloring': 'colouring',
 'colorize': 'colourize',
 'colorized': 'colourized',
 'colorizes': 'colourizes',
 'colorizing': 'colourizing',
 'colorless': 'colourless',
 'colors': 'colours',
 'commercialize': 'commercialise',
 'commercialized': 'commercialised',
 'commercializes': 'commercialises',
 'commercializing': 'commercialising',
 'compartmentalize': 'compartmentalise',
 'compartmentalized': 'compartmentalised',
 'compartmentalizes': 'compartmentalises',
 'compartmentalizing': 'compartmentalising',
 'computerize': 'computerise',
 'computerized': 'computerised',
 'computerizes': 'computerises',
 'computerizing': 'computerising',
 'conceptualize': 'conceptualise',
 'conceptualized': 'conceptualised',
 'conceptualizes': 'conceptualises',
 'conceptualizing': 'conceptualising',
 'connection': 'connexion',
 'connections': 'connexions',
 'contextualize': 'contextualise',
 'contextualized': 'contextualised',
 'contextualizes': 'contextualises',
 'contextualizing': 'contextualising',
 'councilor': 'councillor',
 'councilors': 'councillors',
 'counseled': 'counselled',
 'counseling': 'counselling',
 'counselor': 'counsellor',
 'counselors': 'counsellors',
 'cozier': 'cosier',
 'cozies': 'cosies',
 'coziest': 'cosiest',
 'cozily': 'cosily',
 'coziness': 'cosiness',
 'cozy': 'cosy',
 'crenelated': 'crenellated',
 'criminalize': 'criminalise',
 'criminalized': 'criminalised',
 'criminalizes': 'criminalises',
 'criminalizing': 'criminalising',
 'criticize': 'criticise',
 'criticized': 'criticised',
 'criticizes': 'criticises',
 'criticizing': 'criticising',
 'crueler': 'crueller',
 'cruelest': 'cruellest',
 'crystallization': 'crystallisation',
 'crystallize': 'crystallise',
 'crystallized': 'crystallised',
 'crystallizes': 'crystallises',
 'crystallizing': 'crystallising',
 'cudgeled': 'cudgelled',
 'cudgeling': 'cudgelling',
 'customize': 'customise',
 'customized': 'customised',
 'customizes': 'customises',
 'customizing': 'customising',
 'decentralization': 'decentralisation',
 'decentralize': 'decentralise',
 'decentralized': 'decentralised',
 'decentralizes': 'decentralises',
 'decentralizing': 'decentralising',
 'decriminalization': 'decriminalisation',
 'decriminalize': 'decriminalise',
 'decriminalized': 'decriminalised',
 'decriminalizes': 'decriminalises',
 'decriminalizing': 'decriminalising',
 'defense': 'defence',
 'defenseless': 'defenceless',
 'defenses': 'defences',
 'dehumanization': 'dehumanisation',
 'dehumanize': 'dehumanise',
 'dehumanized': 'dehumanised',
 'dehumanizes': 'dehumanises',
 'dehumanizing': 'dehumanising',
 'demeanor': 'demeanour',
 'demilitarization': 'demilitarisation',
 'demilitarize': 'demilitarise',
 'demilitarized': 'demilitarised',
 'demilitarizes': 'demilitarises',
 'demilitarizing': 'demilitarising',
 'demobilization': 'demobilisation',
 'demobilize': 'demobilise',
 'demobilized': 'demobilised',
 'demobilizes': 'demobilises',
 'demobilizing': 'demobilising',
 'democratization': 'democratisation',
 'democratize': 'democratise',
 'democratized': 'democratised',
 'democratizes': 'democratises',
 'democratizing': 'democratising',
 'demonize': 'demonise',
 'demonized': 'demonised',
 'demonizes': 'demonises',
 'demonizing': 'demonising',
 'demoralization': 'demoralisation',
 'demoralize': 'demoralise',
 'demoralized': 'demoralised',
 'demoralizes': 'demoralises',
 'demoralizing': 'demoralising',
 'denationalization': 'denationalisation',
 'denationalize': 'denationalise',
 'denationalized': 'denationalised',
 'denationalizes': 'denationalises',
 'denationalizing': 'denationalising',
 'deodorize': 'deodorise',
 'deodorized': 'deodorised',
 'deodorizes': 'deodorises',
 'deodorizing': 'deodorising',
 'depersonalize': 'depersonalise',
 'depersonalized': 'depersonalised',
 'depersonalizes': 'depersonalises',
 'depersonalizing': 'depersonalising',
 'deputize': 'deputise',
 'deputized': 'deputised',
 'deputizes': 'deputises',
 'deputizing': 'deputising',
 'desensitization': 'desensitisation',
 'desensitize': 'desensitise',
 'desensitized': 'desensitised',
 'desensitizes': 'desensitises',
 'desensitizing': 'desensitising',
 'destabilization': 'destabilisation',
 'destabilize': 'destabilise',
 'destabilized': 'destabilised',
 'destabilizes': 'destabilises',
 'destabilizing': 'destabilising',
 'dialed': 'dialled',
 'dialing': 'dialling',
 'dialog': 'dialogue',
 'dialogs': 'dialogues',
 'diarrhea': 'diarrhoea',
 'digitize': 'digitise',
 'digitized': 'digitised',
 'digitizes': 'digitises',
 'digitizing': 'digitising',
 'discolor': 'discolour',
 'discolored': 'discoloured',
 'discoloring': 'discolouring',
 'discolors': 'discolours',
 'disemboweled': 'disembowelled',
 'disemboweling': 'disembowelling',
 'disfavor': 'disfavour',
 'disheveled': 'dishevelled',
 'passivizes': 'passivises',
 'passivizing': 'passivising',
 'pasteurization': 'pasteurisation',
 'pasteurize': 'pasteurise',
 'pasteurized': 'pasteurised',
 'pasteurizes': 'pasteurises',
 'pasteurizing': 'pasteurising',
 'patronize': 'patronise',
 'patronized': 'patronised',
 'patronizes': 'patronises',
 'patronizing': 'patronising',
 'patronizingly': 'patronisingly',
 'pedaled': 'pedalled',
 'pedaling': 'pedalling',
 'pederast': 'paederast',
 'pederasts': 'paederasts',
 'pedestrianization': 'pedestrianisation',
 'pedestrianize': 'pedestrianise',
 'pedestrianized': 'pedestrianised',
 'pedestrianizes': 'pedestrianises',
 'pedestrianizing': 'pedestrianising',
 'pediatric': 'paediatric',
 'pediatrician': 'paediatrician',
 'pediatricians': 'paediatricians',
 'pediatrics': 'paediatrics',
 'pedophile': 'paedophile',
 'pedophiles': 'paedophiles',
 'pedophilia': 'paedophilia',
 'penalize': 'penalise',
 'penalized': 'penalised',
 'penalizes': 'penalises',
 'penalizing': 'penalising',
 'penciled': 'pencilled',
 'penciling': 'pencilling',
 'personalize': 'personalise',
 'personalized': 'personalised',
 'personalizes': 'personalises',
 'personalizing': 'personalising',
 'pharmacopeia': 'pharmacopoeia',
 'pharmacopeias': 'pharmacopoeias',
 'philosophize': 'philosophise',
 'philosophized': 'philosophised',
 'philosophizes': 'philosophises',
 'philosophizing': 'philosophising',
 'phony ': 'phoney ',
 'pizzazz': 'pzazz',
 'plagiarize': 'plagiarise',
 'plagiarized': 'plagiarised',
 'plagiarizes': 'plagiarises',
 'plagiarizing': 'plagiarising',
 'plow': 'plough',
 'plowed': 'ploughed',
 'plowing': 'ploughing',
 'plowman': 'ploughman',
 'plowmen': 'ploughmen',
 'plows': 'ploughs',
 'plowshare': 'ploughshare',
 'plowshares': 'ploughshares',
 'polarization': 'polarisation',
 'polarize': 'polarise',
 'polarized': 'polarised',
 'polarizes': 'polarises',
 'polarizing': 'polarising',
 'politicization': 'politicisation',
 'politicize': 'politicise',
 'politicized': 'politicised',
 'politicizes': 'politicises',
 'politicizing': 'politicising',
 'popularization': 'popularisation',
 'popularize': 'popularise',
 'popularized': 'popularised',
 'popularizes': 'popularises',
 'popularizing': 'popularising',
 'pouf': 'pouffe',
 'poufs': 'pouffes',
 'practice': 'practise',
 'practiced': 'practised',
 'practices': 'practises',
 'practicing ': 'practising ',
 'presidium': 'praesidium',
 'presidiums ': 'praesidiums ',
 'pressurization': 'pressurisation',
 'pressurize': 'pressurise',
 'pressurized': 'pressurised',
 'pressurizes': 'pressurises',
 'pressurizing': 'pressurising',
 'pretense': 'pretence',
 'pretenses': 'pretences',
 'primeval': 'primaeval',
 'prioritization': 'prioritisation',
 'prioritize': 'prioritise',
 'prioritized': 'prioritised',
 'prioritizes': 'prioritises',
 'prioritizing': 'prioritising',
 'privatization': 'privatisation',
 'privatizations': 'privatisations',
 'privatize': 'privatise',
 'privatized': 'privatised',
 'privatizes': 'privatises',
 'privatizing': 'privatising',
 'professionalization': 'professionalisation',
 'professionalize': 'professionalise',
 'professionalized': 'professionalised',
 'professionalizes': 'professionalises',
 'professionalizing': 'professionalising',
 'program': 'programme',
 'programs': 'programmes',
 'prolog': 'prologue',
 'prologs': 'prologues',
 'propagandize': 'propagandise',
 'propagandized': 'propagandised',
 'propagandizes': 'propagandises',
 'propagandizing': 'propagandising',
 'proselytize': 'proselytise',
 'proselytized': 'proselytised',
 'proselytizer': 'proselytiser',
 'proselytizers': 'proselytisers',
 'proselytizes': 'proselytises',
 'proselytizing': 'proselytising',
 'psychoanalyze': 'psychoanalyse',
 'psychoanalyzed': 'psychoanalysed',
 'psychoanalyzes': 'psychoanalyses',
 'psychoanalyzing': 'psychoanalysing',
 'publicize': 'publicise',
 'publicized': 'publicised',
 'publicizes': 'publicises',
 'publicizing': 'publicising',
 'pulverization': 'pulverisation',
 'pulverize': 'pulverise',
 'pulverized': 'pulverised',
 'pulverizes': 'pulverises',
 'pulverizing': 'pulverising',
 'pummel': 'pummelled',
 'pummeled': 'pummelling',
 'quarreled': 'quarrelled',
 'quarreling': 'quarrelling',
 'radicalize': 'radicalise',
 'radicalized': 'radicalised',
 'radicalizes': 'radicalises',
 'radicalizing': 'radicalising',
 'rancor': 'rancour',
 'randomize': 'randomise',
 'randomized': 'randomised',
 'randomizes': 'randomises',
 'randomizing': 'randomising',
 'rationalization': 'rationalisation',
 'rationalizations': 'rationalisations',
 'rationalize': 'rationalise',
 'rationalized': 'rationalised',
 'rationalizes': 'rationalises',
 'rationalizing': 'rationalising',
 'raveled': 'ravelled',
 'raveling': 'ravelling',
 'realizable': 'realisable',
 'realization': 'realisation',
 'realizations': 'realisations',
 'realize': 'realise',
 'realized': 'realised',
 'realizes': 'realises',
 'realizing': 'realising',
 'recognizable': 'recognisable',
 'recognizably': 'recognisably',
 'recognizance': 'recognisance',
 'recognize': 'recognise',
 'recognized': 'recognised',
 'recognizes': 'recognises',
 'recognizing': 'recognising',
 'reconnoiter': 'reconnoitre',
 'reconnoitered': 'reconnoitred',
 'reconnoitering': 'reconnoitring',
 'reconnoiters': 'reconnoitres',
 'refueled': 'refuelled',
 'refueling': 'refuelling',
 'regularization': 'regularisation',
 'regularize': 'regularise',
 'regularized': 'regularised',
 'regularizes': 'regularises',
 'regularizing': 'regularising',
 'remodeled': 'remodelled',
 'remodeling': 'remodelling',
 'remold': 'remould',
 'remolded': 'remoulded',
 'remolding': 'remoulding',
 'remolds': 'remoulds',
 'reorganization': 'reorganisation',
 'reorganizations': 'reorganisations',
 'reorganize': 'reorganise',
 'reorganized': 'reorganised',
 'reorganizes': 'reorganises',
 'reorganizing': 'reorganising',
 'reveled': 'revelled',
 'reveler': 'reveller',
 'revelers': 'revellers',
 'reveling': 'revelling',
 'revitalize': 'revitalise',
 'revitalized': 'revitalised',
 'revitalizes': 'revitalises',
 'revitalizing': 'revitalising',
 'revolutionize': 'revolutionise',
 'revolutionized': 'revolutionised',
 'revolutionizes': 'revolutionises',
 'revolutionizing': 'revolutionising',
 'rhapsodize': 'rhapsodise',
 'rhapsodized': 'rhapsodised',
 'rhapsodizes': 'rhapsodises',
 'rhapsodizing': 'rhapsodising',
 'rigor': 'rigour',
 'rigors': 'rigours',
 'ritualized': 'ritualised',
 'rivaled': 'rivalled',
 'rivaling': 'rivalling',
 'romanticize': 'romanticise',
 'romanticized': 'romanticised',
 'romanticizes': 'romanticises',
 'romanticizing': 'romanticising',
 'rumor': 'rumour',
 'rumored': 'rumoured',
 'rumors': 'rumours',
 'saber': 'sabre',
 'sabers': 'sabres',
 'saltpeter': 'saltpetre',
 'sanitize': 'sanitise',
 'sanitized': 'sanitised',
 'sanitizes': 'sanitises',
 'sanitizing': 'sanitising',
 'satirize': 'satirise',
 'satirized': 'satirised',
 'satirizes': 'satirises',
 'satirizing': 'satirising',
 'savior': 'saviour',
 'saviors': 'saviours',
 'savor': 'savour',
 'savored': 'savoured',
 'savories': 'savouries',
 'savoring': 'savouring',
 'savors': 'savours',
 'savory': 'savoury',
 'scandalize': 'scandalise',
 'scandalized': 'scandalised',
 'scandalizes': 'scandalises',
 'scandalizing': 'scandalising',
 'scepter': 'sceptre',
 'scepters': 'sceptres',
 'scrutinize': 'scrutinise',
 'scrutinized': 'scrutinised',
 'scrutinizes': 'scrutinises',
 'scrutinizing': 'scrutinising',
 'secularization': 'secularisation',
 'secularize': 'secularise',
 'secularized': 'secularised',
 'secularizes': 'secularises',
 'secularizing': 'secularising',
 'sensationalize': 'sensationalise',
 'sensationalized': 'sensationalised',
 'sensationalizes': 'sensationalises',
 'sensationalizing': 'sensationalising',
 'sensitize': 'sensitise',
 'sensitized': 'sensitised',
 'sensitizes': 'sensitises',
 'sensitizing': 'sensitising',
 'sentimentalize': 'sentimentalise',
 'sentimentalized': 'sentimentalised',
 'sentimentalizes': 'sentimentalises',
 'sentimentalizing': 'sentimentalising',
 'sepulcher': 'sepulchre',
 'sepulchers ': 'sepulchres',
 'serialization': 'serialisation',
 'serializations': 'serialisations',
 'serialize': 'serialise',
 'serialized': 'serialised',
 'serializes': 'serialises',
 'serializing': 'serialising',
 'sermonize': 'sermonise',
 'sermonized': 'sermonised',
 'sermonizes': 'sermonises',
 'sermonizing': 'sermonising',
 'sheik ': 'sheikh ',
 'shoveled': 'shovelled',
 'shoveling': 'shovelling',
 'shriveled': 'shrivelled',
 'shriveling': 'shrivelling',
 'signaled': 'signalled',
 'signaling': 'signalling',
 'signalize': 'signalise',
 'signalized': 'signalised',
 'signalizes': 'signalises',
 'signalizing': 'signalising',
 'siphon': 'syphon',
 'siphoned': 'syphoned',
 'siphoning': 'syphoning',
 'siphons': 'syphons',
 'skeptic': 'sceptic',
 'skeptical': 'sceptical',
 'skeptically': 'sceptically',
 'skepticism': 'scepticism',
 'skeptics': 'sceptics',
 'smolder': 'smoulder',
 'smoldered': 'smouldered',
 'smoldering': 'smouldering',
 'smolders': 'smoulders',
 'sniveled': 'snivelled',
 'sniveling': 'snivelling',
 'snorkeled': 'snorkelled',
 'snorkeling': 'snorkelling',
 'snowplow': 'snowploughs',
 'socialization': 'socialisation',
 'socialize': 'socialise',
 'socialized': 'socialised',
 'socializes': 'socialises',
 'socializing': 'socialising',
 'sodomize': 'sodomise',
 'sodomized': 'sodomised',
 'sodomizes': 'sodomises',
 'sodomizing': 'sodomising',
 'solemnize': 'solemnise',
 'solemnized': 'solemnised',
 'solemnizes': 'solemnises',
 'solemnizing': 'solemnising',
 'somber': 'sombre',
 'specialization': 'specialisation',
 'specializations': 'specialisations',
 'specialize': 'specialise',
 'specialized': 'specialised',
 'specializes': 'specialises',
 'specializing': 'specialising',
 'specter': 'spectre',
 'specters': 'spectres',
 'spiraled': 'spiralled',
 'spiraling': 'spiralling',
 'splendor': 'splendour',
 'splendors': 'splendours',
 'squirreled': 'squirrelled',
 'squirreling': 'squirrelling',
 'stabilization': 'stabilisation',
 'stabilize': 'stabilise',
 'stabilized': 'stabilised',
 'stabilizer': 'stabiliser',
 'stabilizers': 'stabilisers',
 'stabilizes': 'stabilises',
 'stabilizing': 'stabilising',
 'standardization': 'standardisation',
 'standardize': 'standardise',
 'standardized': 'standardised',
 'standardizes': 'standardises',
 'standardizing': 'standardising',
 'stenciled': 'stencilled',
 'stenciling': 'stencilling',
 'sterilization': 'sterilisation',
 'sterilizations': 'sterilisations',
 'sterilize': 'sterilise',
 'sterilized': 'sterilised',
 'sterilizer': 'steriliser',
 'sterilizers': 'sterilisers',
 'sterilizes': 'sterilises',
 'sterilizing': 'sterilising',
 'stigmatization': 'stigmatisation',
 'stigmatize': 'stigmatise',
 'stigmatized': 'stigmatised',
 'stigmatizes': 'stigmatises',
 'stigmatizing': 'stigmatising',
 'stories': 'storeys',
 'story': 'storey',
 'subsidization': 'subsidisation',
 'subsidize': 'subsidise',
 'subsidized': 'subsidised',
 'subsidizer': 'subsidiser',
 'subsidizers': 'subsidisers',
 'subsidizes': 'subsidises',
 'subsidizing': 'subsidising',
 'succor': 'succour',
 'succored': 'succoured',
 'succoring': 'succouring',
 'succors': 'succours',
 'sulfate': 'sulphate',
 'sulfates': 'sulphates',
 'sulfide': 'sulphide',
 'sulfides': 'sulphides',
 'sulfur': 'sulphur',
 'sulfurous': 'sulphurous'}

UK2US = {uk_word: us_word for us_word, uk_word in US2UK.items()}

def us_to_uk(us_word: str) -> str:
    '''
    :param us_word: A potential US spelt word.
    :returns: A UK spelling of the US word if it is in our `US2UK` dictionary. 
              Else returns the original word.
    '''
    if us_word in US2UK:
        return US2UK[us_word]
    else:
        return us_word

def uk_to_us(uk_word: str) -> str:
    '''
    :param uk_word: A potential UK spelt word.
    :returns: A US spelling of the UK word if it is in our `UK2US` dictionary. 
              Else returns the original word.
    '''
    if uk_word in UK2US:
        return UK2US[uk_word]
    else:
        return uk_word


def read_word_list(word_list_fp: Path) -> List[str]:
    '''
    :param word_list_fp: Path to a word list/lexicon. Expects that each new line
                         contains a word.
    :returns: The word list as a list of strings with UK and US spellings of all 
              words. Also all words will have been lower cased.
    '''
    all_words = set()
    with word_list_fp.open('r') as word_list:
        for line in word_list:
            line = line.strip()
            if line is None:
                continue
            word = line.lower()
            uk_version = us_to_uk(word)
            us_version = uk_to_us(word)
            all_words.add(word)
            all_words.add(uk_version)
            all_words.add(us_version)
    return list(all_words)

def download_and_read_word_list(word_list_name: str) -> List[str]:
    '''
    Names of word lists supported:
    
    1. 'L&M pos'
    2. 'L&M neg'
    3. 'HEN 08 pos'
    4. 'HEN 08 neg'
    5. 'HEN 06 pos'
    6. 'HEN 06 neg'
    7. 'MW TYPE INT'
    8. 'MW TYPE EXT'
    9. 'Dikoli_2016'
    10. 'MW_ALL'
    11. 'MW_50'
    12. 'ZA_2015'

    
    :param word_list_name: Name of the word list you want to download.
    :returns: The downloaded word list as a List of Strings lower cased and with 
              US to UK and UK to US equivalent words included. The US to UK 
              word lists are derived from:
              http://www.tysto.com/uk-us-spelling-list.html
    '''
    cache_dir = Path(Path.home(), '.pea_classification').resolve()
    list_name_url = {'L&M pos': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/LMpos.txt?inline=false',
                     'L&M neg': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/LMneg.txt?inline=false',
                     'HEN 08 pos': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/HenryPos2008.txt?inline=false',
                     'HEN 08 neg': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/HenryNeg2008.txt?inline=false',
                     'HEN 06 pos': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/HenryPos2006.txt?inline=false',
                     'HEN 06 neg': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/sentiment/HenryNeg2006.txt?inline=false',
                     'MW TYPE INT': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution_type/InternalAtt.txt?inline=false',
                     'MW TYPE EXT': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution_type/ExternalAtt.txt?inline=false',
                     'Dikoli_2016': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution/DIKOLLI.txt?inline=false',
                     'MW_ALL': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution/MW_ALL.txt?inline=false',
                     'MW_50': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution/MW_50.txt?inline=false',
                     'ZA_2015': 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/word_lists/attribution/ZA.txt?inline=false'}
    return read_word_list(_get_from_cache(list_name_url[word_list_name], cache_dir))