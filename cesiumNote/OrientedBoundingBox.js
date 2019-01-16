// 相对切片中心点的偏移

var centerOffset=scratchOffset;
centerOffset.x=(minimumX+maximumX)/2.0;
centerOffset.y=(minimumT+maximumY)/2.0;
centerOffset.z=(minimumZ+maximumZ)/2.0;

//相对一个2*2*2的正方体的缩放比
var scale=scratchScale;
scale.x=(maximumX-minimumX)/2.0;
scale.y=(maximumY-minimumY)/2.0;
scale.z=(maximumZ-minimumZ)/2.0;