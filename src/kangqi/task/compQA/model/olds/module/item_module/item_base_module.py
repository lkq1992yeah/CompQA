class ItemBaseModule(object):

    def __init__(self, item_max_len, dim_wd_emb, dim_item_hidden):
        self.item_max_len = item_max_len
        self.dim_wd_emb = dim_wd_emb
        self.dim_item_hidden = dim_item_hidden

    # Input:
    #   item_wd_embedding: (batch, item_max_len, dim_wd_emb)
    #   item_len: (batch, ) as int32
    # Output:
    #   item_wd_hidden: (batch, dim_item_hidden)
    def forward(self, item_wd_embedding, item_len, reuse=None):
        pass
