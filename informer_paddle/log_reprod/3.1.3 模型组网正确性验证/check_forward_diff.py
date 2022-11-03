from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    # init
    diff_helper = ReprodDiffHelper()
    # forward-check model - failed
    torch_info = diff_helper.load_info("./forward_out_torch.npy")
    paddle_info = diff_helper.load_info("./forward_out_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")

    # forward-check enc_embedding - failed
    torch_info = diff_helper.load_info("./forward_enc_out_torch.npy")
    paddle_info = diff_helper.load_info("./forward_enc_out_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")

    # forward-check enc_embedding-TokenEmbedding
    torch_info = diff_helper.load_info("./forward_enc_tokenembedding_out_torch.npy")
    paddle_info = diff_helper.load_info("./forward_enc_tokenembedding_out_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")


