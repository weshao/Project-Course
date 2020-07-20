#!/usr/bin/env bash
echo "Step 1 CIHP: save segmentations to output/segment"
cd CIHP_PGN
python test_pgn.py

echo "Step 2 Openpose: get keypoints"
cd ..
cd openpose
./build/examples/openpose/openpose.bin --image_dir ../input/images -write_json ../output/keypoints --model_pose BODY_25  --write_images ../output/skeleton --face

echo "Step 3 Octopus: get pkl file fot texture and obj file for 3D model"
cd ..
cd octopus
python infer_single.py sample ../output/segment ../output/keypoints --out_dir ../output/obj

echo "Step 4 Semantic_human_texture_stitching"
cd ..
cd semantic_human_texture_stitching
echo "1: make unwraps..."
mkdir -p ../output/unwraps
python step1_make_unwraps.py ../output/pkl/frame_data.pkl ../input/images ../output/segment ../output/unwraps
echo "2: make priors..."
python step2_segm_vote_gmm.py ../output/unwraps ../output/stitch/segm.png ../output/stitch/gmm.pkl
echo "3: stitch texture..."
python step3_stitch_texture.py ../output/unwraps ../output/stitch/segm.png ../output/stitch/gmm.pkl ../output/texture.jpg
