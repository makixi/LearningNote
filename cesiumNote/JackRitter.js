//the same as naive method
//..............
//..........


//if a point lies outside
if(oldCenterToPointSquard>radisSquard){
    var oldCenterToPoint=Math.sqrt(oldCenterToPointSquard);

    //calculate new radius to include the point that lies outside
    ritterRadius=(ritterRadius+oldCenterToPoint)*0.5;
    radiusSqared=ritterRadius*ritterRadius;

    //calculate center of new Ritter sphere
    var oldtoNew=oldCenterToPoint-ritterRadius;
    ritterCenter.x=(ritterRadius*ritterCenter.x+oldtoNew*currentPox.x)/oldCenterToPoint;
    ritterCenter.y=(ritterRadius*ritterCenter.y+oldtoNew*currentPox.y)/oldCenterToPoint;
    ritterCenter.z=(ritterRadius*ritterCenter.z+oldtoNew*currentPox.z)/oldCenterToPoint;
}