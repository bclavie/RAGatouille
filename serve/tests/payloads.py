search_query = {
  "query": "egypt",
  "index_name": "string",
  "k": 10,
  "force_fast": True,
  "zero_index_ranks": False
}

index_documents = {
  "collection": [
    "Ancient Egyptian literature was written in the Egyptian language from Ancient Egypt's pharaonic period until the end of Roman domination. Along with Sumerian literature, it is considered the world's earliest literature. Writing in Ancient Egypt (sample pictured) first appeared in the late 4th millennium BC. By the Old Kingdom, literary works included funerary texts, epistles and letters, religious hymns and poems, and commemorative autobiographical texts. Middle Egyptian, the spoken language of the Middle Kingdom, became a classical language preserving a narrative Egyptian literature during the New Kingdom, when Late Egyptian first appeared in writing. Scribes of the New Kingdom canonized and copied many literary texts written in Middle Egyptian, which remained the language used for oral readings of sacred hieroglyphic texts. Ancient Egyptian literature has been preserved on papyrus scrolls and packets, limestone and ceramic ostraca, wooden writing boards, monumental stone edifices, and coffins.",
    "The redcurrant (Ribes rubrum) is a deciduous shrub in the gooseberry family, Grossulariaceae, which is native to western Europe. The plant normally grows to a height of up to one metre (3 ft), with its leaves arranged spirally on the stems. The flowers are inconspicuous yellow-green, maturing into bright red translucent edible berries. An established redcurrant bush can produce 3 to 4 kilograms (7 to 9 lb) of berries from mid- to late summer. The species is widely cultivated, with the berries known for their tart flavor, a characteristic provided by a relatively high content of organic acids and mixed polyphenols. This photograph of a bunch of redcurrant berries was focus-stacked from 15 separate images."
  ],
  "document_ids": [
    "1", "2"
  ],
  "document_metadatas": [
    {"name":"Wikipedia"},
    {"name":"Wikipedia"}
  ],
  "index_name": "string",
  "overwrite_index": True,
  "max_document_length": 256,
  "split_documents": True
}

add_documents = {
  "new_collection": [
    "1287 – Wareru created the Hanthawaddy Kingdom in today's Lower Burma and declared himself king following the collapse of the Pagan Empire. 1661 – Two years after his death, Oliver Cromwell's remains were exhumed for a posthumous execution and his head was placed on a spike above Westminster Hall in London, where it remained until 1685. 1945 – World War II: Allied forces liberated more than 500 prisoners of war (pictured) from a Japanese POW camp near Cabanatuan in the Philippines. 2020 – The World Health Organization declared the COVID-19 pandemic to be a public health emergency of international concern."
  ],
  "new_document_ids": [
    "3"
  ],
  "new_document_metadatas": [
    {"name":"Wikipedia"}
  ],
  "index_name": "string",
  "split_documents": True
}

delete_documents = {
  "document_ids": [
    "1"
  ],
  "index_name": "string"
}

rerank = {
  "query": "berry",
  "documents": [
    "Ancient Egyptian literature was written in the Egyptian language from Ancient Egypt's pharaonic period until the end of Roman domination. Along with Sumerian literature, it is considered the world's earliest literature. Writing in Ancient Egypt (sample pictured) first appeared in the late 4th millennium BC. By the Old Kingdom, literary works included funerary texts, epistles and letters, religious hymns and poems, and commemorative autobiographical texts. Middle Egyptian, the spoken language of the Middle Kingdom, became a classical language preserving a narrative Egyptian literature during the New Kingdom, when Late Egyptian first appeared in writing. Scribes of the New Kingdom canonized and copied many literary texts written in Middle Egyptian, which remained the language used for oral readings of sacred hieroglyphic texts. Ancient Egyptian literature has been preserved on papyrus scrolls and packets, limestone and ceramic ostraca, wooden writing boards, monumental stone edifices, and coffins.",
"The redcurrant (Ribes rubrum) is a deciduous shrub in the gooseberry family, Grossulariaceae, which is native to western Europe. The plant normally grows to a height of up to one metre (3 ft), with its leaves arranged spirally on the stems. The flowers are inconspicuous yellow-green, maturing into bright red translucent edible berries. An established redcurrant bush can produce 3 to 4 kilograms (7 to 9 lb) of berries from mid- to late summer. The species is widely cultivated, with the berries known for their tart flavor, a characteristic provided by a relatively high content of organic acids and mixed polyphenols. This photograph of a bunch of redcurrant berries was focus-stacked from 15 separate images."
  ],
  "k": 2,
  "zero_index_ranks": False,
  "bsize": 64
}

encode = {
  "documents": [
    "Ancient Egyptian literature was written in the Egyptian language from Ancient Egypt's pharaonic period until the end of Roman domination. Along with Sumerian literature, it is considered the world's earliest literature. Writing in Ancient Egypt (sample pictured) first appeared in the late 4th millennium BC. By the Old Kingdom, literary works included funerary texts, epistles and letters, religious hymns and poems, and commemorative autobiographical texts. Middle Egyptian, the spoken language of the Middle Kingdom, became a classical language preserving a narrative Egyptian literature during the New Kingdom, when Late Egyptian first appeared in writing. Scribes of the New Kingdom canonized and copied many literary texts written in Middle Egyptian, which remained the language used for oral readings of sacred hieroglyphic texts. Ancient Egyptian literature has been preserved on papyrus scrolls and packets, limestone and ceramic ostraca, wooden writing boards, monumental stone edifices, and coffins."
  ],
  "bsize": 32,
  "document_metadatas": [
    {"name":"wikipedia"}
  ],
  "verbose": True,
  "max_document_length": "auto"
}

search_encoded = {
  "query": "egypt",
  "k": 10,
  "bsize": 32
}

clear_encoded = {
  "force": True
}
