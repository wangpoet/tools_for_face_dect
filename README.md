# tools_for_face_dect

Implementation of face detection  using dlib.


# Usage

## Environment
dlib+python+windows(or linux)


The structure of my dataset:

<pre>
src/
├── people_folderA
│ �?├── img1.jpg
│ �?├── img2.jpg
│ �?├── imgN.jpg
└── people_folderB
</pre>


## Usage of Code

#### crop_the_face.py
	python crop_the_face.py


#### result:
    In the dect_folder , you will find the picture after croping,and  list contains the location information of the human face,see below.

```
./dest/1/002.jpg,44,44,200,200
./dest/1/004.jpg,51,66,181,196
./dest/1/016.jpg,66,66,174,175
./dest/1/017.jpg,51,66,181,196
./dest/2/008.jpg,65,66,195,196
./dest/2/009.jpg,44,62,200,217
```

