# zipnerf-nerfstudio

currently borked. hoping to be less borked, extenal collaborators welcome, feel free to make a PR and / or an issue :)

Notes:
Multisampling / downweighting is now implemented. Looks good, but more sanity checks are welcome.

Things yet to be implemented:
Appending of featurized multisample weights seems to break things, this may be because of the stopgrad that removes gradients to the feature grid itself, but I'm not sure since that stopgrad is specified in the paper.
Zipnerf proposal loss seems to get sub-par results compared to the mipnerf360 proposal loss, so for now we still have the mipnerf360 proposal loss.
Proper batch size / learning rate / network size as specified by the zipnerf paper.

Misc info:
Aiming still to have this run on 10GB vram, by using tinycudann's marginally smaller MLPs and decreasing the learning rate and batch size to be more reasonable.
Basing a lot of this off https://github.com/SuLvXiangXin/zipnerf-pytorch, just trying to get it further integrated with nerfstudio so we can officially list zipnerf as an external method for nerfstudio.
