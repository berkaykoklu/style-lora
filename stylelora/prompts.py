"""The subject pool, in two languages.

Subjects only: no prompt here names a style. A prompt that named one would let
the base model produce it too, and the adapter's contribution would vanish into
the wording -- the same mistake as letting a caption carry the style, arriving
through a different door.

Every one of these is something both a Japanese woodblock print and a Dutch oil
painting would depict. That is what makes the pair measurable: the styles differ
in *how* they depict, so a prompt only one of them could attempt would confound
style with subject.

The English goes to the model; the Turkish goes on the rating card, because the
person judging whether the prompt survived has to be able to read it.
"""

from __future__ import annotations

PAIRS: tuple[tuple[str, str], ...] = (
    ('a woman holding a lantern', 'elinde fener tutan bir kadın'),
    ('a knight standing in a doorway', 'kapı eşiğinde duran bir şövalye'),
    ('a fox in a forest clearing', 'orman açıklığında bir tilki'),
    ('a city street after rain', 'yağmurdan sonra bir şehir sokağı'),
    ('a young man reading a letter', 'mektup okuyan genç bir adam'),
    ('a harbour at sunrise', 'gün doğumunda bir liman'),
    ('a cat asleep on a windowsill', 'pencere kenarında uyuyan bir kedi'),
    ('two people talking at a table', 'masada konuşan iki kişi'),
    ('a horse in an open field', 'açık bir tarlada bir at'),
    ('a staircase in an empty hall', 'boş bir salonda bir merdiven'),
    ('a bowl of fruit on a cloth', 'kumaş üzerinde bir kâse meyve'),
    ('a traveller on a mountain path', 'dağ yolunda bir yolcu'),
    ('a woman combing her long hair', 'uzun saçlarını tarayan bir kadın'),
    ('a bridge over a river in the rain', 'yağmurda nehrin üzerinde bir köprü'),
    ('a fisherman pulling in a net', 'ağını çeken bir balıkçı'),
    ('an old man drinking from a cup', 'fincandan içen yaşlı bir adam'),
    ('a group of travellers resting under a tree', 'ağaç altında dinlenen yolcular'),
    ('a mountain seen across water', 'suyun karşısından görünen bir dağ'),
    ('a woman holding a fan', 'elinde yelpaze tutan bir kadın'),
    ('two warriors facing each other', 'karşı karşıya duran iki savaşçı'),
    ('a child chasing a bird', 'kuş kovalayan bir çocuk'),
    ('a boat on a rough sea', 'dalgalı denizde bir tekne'),
    ('a garden gate at dusk', 'alacakaranlıkta bir bahçe kapısı'),
    ('a street of shops in the evening', 'akşam vakti dükkânlarla dolu bir sokak'),
    ('a musician playing an instrument', 'çalgı çalan bir müzisyen'),
    ('a table set for a meal', 'yemek için kurulmuş bir masa'),
    ('a dog lying by a fire', 'ateşin yanında yatan bir köpek'),
    ('a woman carrying water', 'su taşıyan bir kadın'),
    ('a tree in blossom beside a path', 'patika kenarında çiçek açmış bir ağaç'),
    ('a man asleep in a chair', 'koltukta uyuyan bir adam'),
    ('a man rowing a small boat', 'küçük bir kayık çeken bir adam'),
    ('a woman kneeling beside a stream', 'derenin kenarında diz çökmüş bir kadın'),
    ('two children playing in the snow', 'karda oynayan iki çocuk'),
    ('a heron standing in shallow water', 'sığ suda duran bir balıkçıl'),
    ('a monk walking with a staff', 'asasıyla yürüyen bir keşiş'),
    ('a merchant weighing goods on a scale', 'terazide mal tartan bir tüccar'),
    ('a woman reading by candlelight', 'mum ışığında okuyan bir kadın'),
    ('an empty room with an open window', 'penceresi açık boş bir oda'),
    ('a crowd gathered in a square', 'meydanda toplanmış kalabalık'),
    ('a rider crossing a shallow ford', 'sığ geçitten geçen bir atlı'),
    ('a blacksmith at his forge', 'ocağının başında bir demirci'),
    ('a woman folding cloth', 'kumaş katlayan bir kadın'),
    ('a cat watching a bird through glass', 'camdan kuş izleyen bir kedi'),
    ('a farmer leading an ox', 'öküz güden bir çiftçi'),
    ('a lantern hanging above a doorway', 'kapının üstünde asılı bir fener'),
    ('a group of men playing a game', 'oyun oynayan bir grup adam'),
    ('a woman asleep under a blanket', 'battaniye altında uyuyan bir kadın'),
    ('a pine tree on a rocky slope', 'kayalık yamaçta bir çam ağacı'),
    ('a girl holding a small mirror', 'elinde küçük bir ayna tutan kız'),
    ('a man carrying a bundle on his back', 'sırtında bohça taşıyan bir adam'),
    ('a wave breaking against rocks', 'kayalara çarpan bir dalga'),
    ('a rooftop seen above a wall', 'duvarın üstünden görünen bir çatı'),
    ('a woman pouring from a jug', 'testiden su döken bir kadın'),
    ('an elderly couple sitting together', 'yan yana oturan yaşlı bir çift'),
    ('a hunter with a bow', 'yaylı bir avcı'),
    ('a bowl and a knife on a wooden board', 'tahta üzerinde bir kâse ve bir bıçak'),
    ('a road winding between hills', 'tepeler arasında kıvrılan bir yol'),
    ('a woman washing clothes in a basin', 'leğende çamaşır yıkayan bir kadın'),
    ('a bird perched on a bare branch', 'çıplak dalda tüneyen bir kuş'),
    ('a man sharpening a blade', 'bıçak bileyen bir adam'),
    ('a temple seen through trees', 'ağaçların arasından görünen bir tapınak'),
    ('two women whispering to each other', 'birbirine fısıldaşan iki kadın'),
    ('a horse drinking from a trough', 'yalaktan su içen bir at'),
    ('a ship at anchor in calm water', 'durgun suda demirlemiş bir gemi'),
    ('a stack of books on a desk', 'masanın üstünde bir yığın kitap'),
    ('a child feeding chickens', 'tavuk besleyen bir çocuk'),
    ('a woman standing in a doorway at night', 'gece kapı eşiğinde duran bir kadın'),
    ('a fire burning in a hearth', 'ocakta yanan bir ateş'),
    ('a man counting coins', 'para sayan bir adam'),
    ('a field of grass bending in wind', 'rüzgârda eğilen bir çayır'),
    ('a young woman with flowers in her hair', 'saçında çiçekler olan genç bir kadın'),
    ('a dog watching from a doorway', 'kapı aralığından bakan bir köpek'),
    ('a bridge crowded with people', 'insanlarla dolu bir köprü'),
    ('a man writing at a small table', 'küçük bir masada yazan bir adam'),
    ('a woman looking out over water', 'suya bakan bir kadın'),
    ('a basket of fish on the ground', 'yerde bir sepet balık'),
    ('an old wall with a broken gate', 'kırık kapılı eski bir duvar'),
    ('a musician resting between songs', 'şarkılar arasında dinlenen bir müzisyen'),
    ('a boat pulled up on a beach', 'kumsala çekilmiş bir tekne'),
    ('a woman climbing a set of stairs', 'merdiven çıkan bir kadın'),
    ('a man wearing a wide hat', 'geniş şapkalı bir adam'),
    ('rain falling on a quiet street', 'sessiz bir sokağa yağan yağmur'),
    ('a room lit by a single window', 'tek pencereyle aydınlanan bir oda'),
    ('a woman gathering herbs', 'ot toplayan bir kadın'),
    ('two men carrying a heavy load', 'ağır bir yük taşıyan iki adam'),
    ('a cat curled beside a brazier', 'mangalın yanında kıvrılmış bir kedi'),
    ('a mountain path in fog', 'sisli bir dağ yolu'),
    ('a market stall at dawn', 'şafakta bir pazar tezgâhı'),
    ('a woman holding a child', 'kucağında çocuk tutan bir kadın'),
    ('a man standing at the edge of a cliff', 'uçurumun kenarında duran bir adam'),
    ('a flock of birds over a field', 'tarlanın üstünde bir kuş sürüsü'),
    ('a table with bread and wine', 'ekmek ve şarapla dolu bir masa'),
    ('a woman spinning thread', 'iplik eğiren bir kadın'),
    ('a traveller sheltering from rain', 'yağmurdan korunan bir yolcu'),
    ('a bare tree against the sky', 'gökyüzüne karşı çıplak bir ağaç'),
    ('a man leading a horse through snow', 'karda at çeken bir adam'),
    ('a woman seated with her back turned', 'sırtı dönük oturan bir kadın'),
    ('a boat crossing a wide river', 'geniş bir nehri geçen tekne'),
    ('a lamp on a table beside a book', 'kitabın yanında masadaki bir lamba'),
    ('a garden seen from a window', 'pencereden görünen bir bahçe'),
    ('a man resting against a tree', 'ağaca yaslanmış dinlenen bir adam'),
    ('a woman carrying a basket on her head', 'başında sepet taşıyan bir kadın'),
)

POOL: tuple[str, ...] = tuple(en for en, _ in PAIRS)
TURKISH: dict[str, str] = {en: tr for en, tr in PAIRS}


def turkish(prompt: str) -> str:
    """The card's text. Falls back to the English rather than to nothing -- an
    untranslated prompt is readable; a blank one makes the question
    unanswerable."""
    return TURKISH.get(prompt, prompt)
