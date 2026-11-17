# EDA WaxalNLP (lin, sna, lug)

## train
| lang   |     n |   speakers |   est_hours |   dur_mean |   dur_p95 |   dur_max |   silence |   words_mean |   empty_text |   female | sr                     |
|:-------|------:|-----------:|------------:|-----------:|----------:|----------:|----------:|-------------:|-------------:|---------:|:-----------------------|
| lin    | 14400 |         95 |        71.8 |       17.9 |      27.2 |      34.9 |      0.39 |         27.5 |            1 |     0.57 | {48000: 199, 44100: 1} |
| lug    |  5455 |        334 |        36.2 |       23.9 |      38.7 |      44.1 |      0.25 |         28.9 |            0 |     0.46 | {16000: 200}           |
| sna    | 14109 |        157 |        81.1 |       20.7 |      28.5 |      33.9 |      0.32 |         23.4 |            0 |     0.53 | {48000: 200}           |

## validation
| lang   |    n |   speakers |   est_hours |   dur_mean |   dur_p95 |   dur_max |   silence |   words_mean |   empty_text |   female | sr                     |
|:-------|-----:|-----------:|------------:|-----------:|----------:|----------:|----------:|-------------:|-------------:|---------:|:-----------------------|
| lin    | 1844 |         73 |         9.5 |       18.5 |      28   |      35.4 |      0.41 |         27.4 |            0 |     0.56 | {48000: 199, 44100: 1} |
| lug    |  664 |        252 |         4.6 |       25.1 |      41.8 |      44.1 |      0.27 |         29.5 |            0 |     0.43 | {16000: 200}           |
| sna    | 1727 |        100 |         9.6 |       20   |      28.5 |      32.9 |      0.32 |         23.1 |            0 |     0.53 | {48000: 200}           |
- lin: val speakers also in train = 72/73
- sna: val speakers also in train = 97/100
- lug: val speakers also in train = 247/252

## test
| lang   |    n |   speakers |   est_hours |   dur_mean |   dur_p95 |   dur_max |   silence |   words_mean |   empty_text |   female | sr                     |
|:-------|-----:|-----------:|------------:|-----------:|----------:|----------:|----------:|-------------:|-------------:|---------:|:-----------------------|
| lin    | 1866 |         75 |         9.2 |       17.7 |      25.7 |      32.5 |      0.39 |            0 |         1866 |     0.59 | {48000: 199, 44100: 1} |
| lug    |  638 |        232 |         4.2 |       23.5 |      41.2 |      44.6 |      0.22 |            0 |          638 |     0.46 | {16000: 200}           |
| sna    | 1749 |        115 |         9.7 |       20   |      28.5 |      34.7 |      0.31 |            0 |         1749 |     0.54 | {48000: 200}           |

## Characters (train)

**lin**: 112 distinct characters. Non-ASCII-alphanumeric: `,` U+002C COMMA (11671), `.` U+002E FULL STOP (11568), `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE (1129), `-` U+002D HYPHEN-MINUS (523), `
` U+000A ? (463), `'` U+0027 APOSTROPHE (360), `!` U+0021 EXCLAMATION MARK (237), `è` U+00E8 LATIN SMALL LETTER E WITH GRAVE (90), `î` U+00EE LATIN SMALL LETTER I WITH CIRCUMFLEX (77), `ô` U+00F4 LATIN SMALL LETTER O WITH CIRCUMFLEX (52), `;` U+003B SEMICOLON (47), `â` U+00E2 LATIN SMALL LETTER A WITH CIRCUMFLEX (22), `ñ` U+00F1 LATIN SMALL LETTER N WITH TILDE (20), `à` U+00E0 LATIN SMALL LETTER A WITH GRAVE (20), `"` U+0022 QUOTATION MARK (20), `ê` U+00EA LATIN SMALL LETTER E WITH CIRCUMFLEX (15), `ï` U+00EF LATIN SMALL LETTER I WITH DIAERESIS (11), `ç` U+00E7 LATIN SMALL LETTER C WITH CEDILLA (10), `?` U+003F QUESTION MARK (10), `É` U+00C9 LATIN CAPITAL LETTER E WITH ACUTE (6), `ì` U+00EC LATIN SMALL LETTER I WITH GRAVE (6), `ù` U+00F9 LATIN SMALL LETTER U WITH GRAVE (5), ` ` U+00A0 NO-BREAK SPACE (4), `&` U+0026 AMPERSAND (3), `/` U+002F SOLIDUS (3), `û` U+00FB LATIN SMALL LETTER U WITH CIRCUMFLEX (2), `«` U+00AB LEFT-POINTING DOUBLE ANGLE QUOTATION MARK (2), `»` U+00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (2), `:` U+003A COLON (2), `⭐` U+2B50 WHITE MEDIUM STAR (2), `️` U+FE0F VARIATION SELECTOR-16 (2), `ĺ` U+013A LATIN SMALL LETTER L WITH ACUTE (2), `ü` U+00FC LATIN SMALL LETTER U WITH DIAERESIS (2), `=` U+003D EQUALS SIGN (2), `í` U+00ED LATIN SMALL LETTER I WITH ACUTE (1), `ĝ` U+011D LATIN SMALL LETTER G WITH CIRCUMFLEX (1), `þ` U+00FE LATIN SMALL LETTER THORN (1), ``` U+0060 GRAVE ACCENT (1), `œ` U+0153 LATIN SMALL LIGATURE OE (1), `Ķ` U+0136 LATIN CAPITAL LETTER K WITH CEDILLA (1)
  after normalize(): " '0123456789abcdefghijklmnopqrstuvwxyzàáâçèéêìíîïñòóôùûüþĝķĺœ"

**lug**: 70 distinct characters. Non-ASCII-alphanumeric: `'` U+0027 APOSTROPHE (14600), `.` U+002E FULL STOP (10634), `,` U+002C COMMA (6593), `ŋ` U+014B LATIN SMALL LETTER ENG (427), `;` U+003B SEMICOLON (322), `ᵑ` U+1D51 MODIFIER LETTER SMALL ENG (50), `-` U+002D HYPHEN-MINUS (47), `:` U+003A COLON (38), `!` U+0021 EXCLAMATION MARK (32), `
` U+000A ? (27), `?` U+003F QUESTION MARK (23), `"` U+0022 QUOTATION MARK (10), `>` U+003E GREATER-THAN SIGN (3), `<` U+003C LESS-THAN SIGN (2), `à` U+00E0 LATIN SMALL LETTER A WITH GRAVE (1)
  after normalize(): " '0146abcdefghijklmnopqrstuvwxyzàŋᵑ"

**sna**: 85 distinct characters. Non-ASCII-alphanumeric: `.` U+002E FULL STOP (37892), `,` U+002C COMMA (7174), `-` U+002D HYPHEN-MINUS (521), `'` U+0027 APOSTROPHE (293), `"` U+0022 QUOTATION MARK (292), `
` U+000A ? (238), `(` U+0028 LEFT PARENTHESIS (101), `)` U+0029 RIGHT PARENTHESIS (101), `í` U+00ED LATIN SMALL LETTER I WITH ACUTE (53), `ú` U+00FA LATIN SMALL LETTER U WITH ACUTE (16), `à` U+00E0 LATIN SMALL LETTER A WITH GRAVE (12), `&` U+0026 AMPERSAND (8), `ñ` U+00F1 LATIN SMALL LETTER N WITH TILDE (6), `ó` U+00F3 LATIN SMALL LETTER O WITH ACUTE (6), ``` U+0060 GRAVE ACCENT (5), `á` U+00E1 LATIN SMALL LETTER A WITH ACUTE (4), `;` U+003B SEMICOLON (2), `/` U+002F SOLIDUS (1), `ò` U+00F2 LATIN SMALL LETTER O WITH GRAVE (1), `!` U+0021 EXCLAMATION MARK (1), `×` U+00D7 MULTIPLICATION SIGN (1), `ā` U+0101 LATIN SMALL LETTER A WITH MACRON (1), `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE (1)
  after normalize(): " '012345789abcdefghijklmnopqrstuvwxyzàáéíñòóúā"

## Sample transcriptions (train)
- lin: Eza ba farde ya langi ya mosaka ya langi ya lilala ya langi ya bozenga ya langi ya motani na ya langi ya pondu na ya pembe.
- lin: Photo elakisi biso mwa place batekisaka Biloko Yako lata ya mibali pe ya basi 
- lin: ndenge mwasi azolakisa biso suki na ye suki na ye basanole yango bien kitoko basalaango backhall ye azotalana se po alakisa biso suki alati elamba ya langi ya motane ya moke
- lin: Ba soda mibale bazo kende na nzela ba lati bilaba na bango ya kisoda pembeni na bango motuka monene eza ya bisi eza ya pembe liboso na bango ba nzete eza na katikati ya nzela ba mwinda eza ya milai. 
- lin: Awa tomoni balakisi biso eloko babengi stationaris to bapetery kasi eloko baza konakisa biso awa ezali babuku te me balakisi biso bakalamu oyo ya kokomela tomoni ezali na malangi mingi eza na balangi ya mutane bululu mosaka moindo nde ya wana baza konakisa biso awa.
- sna: Imba irikuvakwa isati yapera, irikuvakwa nezvidhinha zvesamende. Panofanhirwa kunge paine fafitera parikurongwa zvidhinha . Yakakomberedzwa nemahuswa arikuoneka. Pedyo nayo pane imwe Imba  irikuonekawo ye simendi ne zvidhinha. Uyezve pamusoro pedyo nemba pane gomo riripo. Rine zvimiti nechi pillar chemagetsi chakareba.
- sna: Vanhu vaviri varikufamba. Murume nemukadzi. Munhukadzi akabereka mwana kumusana. Parutivi pane chivakwa chakakura chineruvara ruchena uye nematombo matema anopenya.
- sna: Ndiro ine gomba yakagadzikwa pamusoro pejira jena riri pamusoro petafura ine ruvara ruchena. Ndiro iyi ine ruvara ruchena zvakare uye makapakurirwa maguru namatumbu.
- sna: Masipunu anoshandiswa mukubika nemukubheka ane ruvara rwakasiyana siyana. Mamwe ane ruvara rwebhuruu mamwe egirinhi amwe zvepingi. Akakataniswa pamusoro nekukura kwawo.
- sna: Masora akaoma ari pedyo nenzira, nedzimba dzisina mafafitera, dzine marata esirivha dzine miti iri kuseri.
- lug: Nno waliwo ekiyumba ekyazimbibwa n'embaawo era nga kikomereddwako n'amabaati waggulu nga kyazimbibwa wansi w'omuti era nga ne ku muti kuliko  waya evaako ng'egenda ku muti omulala era nga waya erabiddwako ng'eyanikibbwako engoye ez'abaana abato era nga weetooloddwa emiti n'ebimuli ebya green. 
- lug: Ekinyonyi kiikyo nga kiri ku kiswa ekyafuluka. Kirina kkala ziizo ziri ssatu: kirina kkala emmyufu, kirina kkala eya enzirugavu, kirina kkala eya bbululu.
- lug: Abavubuka babiri nga bayimiridde mu miryango egy'ennyumba ey'ettaka ssaako n'amaba, ennyumba eno waliwo ebimera eby'enjawulo era nga bano bambadde enkoofiira ng'omu ku bo aku yambadde ekijirkooti eky'emikono emiwanvu era ng'akutte n'omuggo ng'omulala ayambadde akajirikooto akaliko langi enjeru era nga n'emikono ssayidi emu nzirugavu.
- lug: Omusajja ng'ayambadde essaati ey'omweso era ng'atadde gaalubindi ku maaso ge. Ku mutwe gwe ataddeko enkofiira eya ppulasitiika. Mu mikono gye mulimu enkampa era ng'asibye n'essaawa, yali ng'ayina by'atereeza ku kisenge oboolyawo byali byamasannyalaze.
- lug: Eno ngabi eri mu kyereerezi. Erina langi eya kitaka ne langi enjeru; erina n'amayembe.
