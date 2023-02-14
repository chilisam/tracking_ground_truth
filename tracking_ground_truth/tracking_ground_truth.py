import math
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import message_filters
from sensor_msgs.msg import NavSatFix
from localization_msgs.msg import GnssOdom
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point, Quaternion
from msg_ros.msg import BoundingBoxArray, BoundingBox
import tf_transformations as tf
from geometry_msgs.msg import Transform

import struct
import std_msgs.msg as std_msgs



_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


def point_cloud(points, parent_frame, field_names="xyzv"):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    
    data = points.astype(dtype).tobytes()

    itemsize = np.dtype(dtype).itemsize
    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(field_names)]

    header = std_msgs.Header(frame_id=parent_frame)

    n_fields = len(field_names)
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * n_fields),
        row_step=(itemsize * n_fields * points.shape[0]),
        data=data
    )

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def convert_WGS84_To_UTM(lat1, long1):  # coordinates conversions
    zone = 33  # for berlin
    a = 6378137
    b = 6356752.3142
    c = (a**2) / b
    alpha = (a - b) / a
    lat = lat1 * (math.pi) / 180
    long2 = long1 * (math.pi) / 180
    long0 = ((zone * 6) - 183) * (math.pi) / 180
    k0 = 0.9996
    e = math.sqrt(((a**2) - (b**2))) / a
    e_prim = (
        math.sqrt(((a**2) - (b**2))) / b
    )  # approximately. The quantity e' only occurs in even powers so it need only be calculated as e'2.
    p = (
        long2 - long0
    )  # in radians (This differs from the treatment in the Army reference)
    A = (math.cos(lat)) * (math.sin(p))
    epsilon = (math.log((1 + A) / (1 - A))) / 2
    mu = (math.atan(math.tan(lat) / math.cos(p))) - lat
    phi = c / (math.sqrt((1 + (e_prim**2) * (math.cos(lat)) ** 2))) * 0.9996
    omega = (((e_prim) ** 2) / 2) * (epsilon**2) * (math.cos(lat)) ** 2
    A1 = math.sin(2 * lat)
    A2 = A1 * (math.cos(lat)) ** 2
    J2 = lat + A1 / 2
    J4 = (3 * J2 + A2) / 4
    J6 = (5 * J4 + A2 * (math.cos(lat)) ** 2) / 3
    beta1 = 3 * (e_prim**2) / 4
    beta2 = 5 * (beta1**2) / 3
    beta3 = 35 * (beta1**3) / 27
    Bo = 0.9996 * c * (lat - beta1 * J2 + beta2 * J4 - beta3 * J6)
    X = epsilon * phi * (1 + omega / 3) + 500000
    Y = mu * phi * (1 + omega) + Bo
    return X, Y


# from stack_interfaces.msg import GnssOdom, BoundingBox, BoundingBoxArray
import utm
def homogeneous_inverse(hom_matrix):
    inv_rot = np.linalg.inv(hom_matrix[:3, :3])
    inv_translation = -inv_rot@np.array(hom_matrix[:3, 3])
    inverted_matrix = np.identity(4)
    inverted_matrix[:3, :3] = inv_rot
    inverted_matrix[:3, 3] = inv_translation
    return inverted_matrix


class TrackingGT(Node):
    def __init__(self):
        super().__init__('tracking_gt')
        self.target_frame = "map"
        self.ref_lat = 52.55754329939843
        self.ref_long = 13.281288759978164
        # self.ref_x, self.ref_y, _, _ = utm.from_latlon(self.ref_lat, self.ref_long)
        self.ref_x, self.ref_y  = convert_WGS84_To_UTM(self.ref_lat, self.ref_long)
        # create publishers
        # self.objects_publisher_ = self.create_publisher(ActorInformationArray, '/obj_vel', 10)
        self.object_1_pc_publisher_ = self.create_publisher(PointCloud2, '/object_gt', 10)
        self.gt_bboxes_publisher_  = self.create_publisher(BoundingBoxArray, '/perception/tracking/gt_bboxes', 10)
        self.pred_bboxes_publisher_  = self.create_publisher(BoundingBoxArray, '/perception/tracking/pred_bboxes', 10)


        ego_gps_location_subscriber = message_filters.Subscriber(self, GnssOdom, "/localization/filtered_output")
        object_location_subscirber = message_filters.Subscriber(self, NavSatFix, "/perception/eqv/gnss")
        bboxes_subscriber = message_filters.Subscriber(self, BoundingBoxArray, "/perception/tracking/bboxes")

        
        # ts = message_filters.ApproximateTimeSynchronizer([ego_gps_location_subscriber, object_location_subscirber, bboxes_subscriber], 200, 0.1)
        ts = message_filters.ApproximateTimeSynchronizer([ego_gps_location_subscriber, object_location_subscirber], 200, 0.1)
        ts.registerCallback(self.transform_callback)
        print("tracking ground truth initialized.")
        self.prev_x = 0
        self.prev_y = 0
        self.first = True

    def transform_callback(self, ego_gps_location, object_status):
        # object's absolute coordinates
        obj_x_abs, obj_y_abs = convert_WGS84_To_UTM(object_status.latitude, object_status.longitude)
        # object's coordinates relative to reference point
        obj_x_ref, obj_y_ref = obj_x_abs - self.ref_x, obj_y_abs - self.ref_y

        # object's velocity in reference coordinates (assuming yaw = velocit direction)
        vx = obj_x_ref - self.prev_x
        vy = obj_y_ref - self.prev_y
        self.prev_x = obj_x_ref
        self.prev_y = obj_y_ref
        if self.first:
            ref_obj_yaw = 0
            self.first = False
        else:
            ref_obj_yaw = math.atan2(vy, vx)
        # ref_obj_yaw = 0

        # ref->obj 
        ref_obj_t = tf.euler_matrix(0, 0, ref_obj_yaw)
        ref_obj_t[:, 3] = [obj_x_ref, obj_y_ref, 0, 1]

        # ref->gps 
        gps_pos =  ego_gps_location.filtered_odometry.pose.pose.position
        gps_orientation = ego_gps_location.filtered_odometry.pose.pose.orientation
        ref_gps_t = tf.quaternion_matrix([gps_orientation.x, gps_orientation.y, gps_orientation.z, gps_orientation.w])
        ref_gps_t[:, 3] = [gps_pos.x, gps_pos.y, gps_pos.z, 1]
        ref_gps_rpy = tf.euler_from_quaternion([gps_orientation.x, gps_orientation.y, gps_orientation.z,gps_orientation.w])

        # gps->ref 
        gps_ref_t = homogeneous_inverse(ref_gps_t)

        # gps->top_lidar 
        gps_lidar_rotation = [0, 0, 0, 1] #7071068, 0.7071068]
        gps_lidar_translation = [1.7, 0, 0, 1] # homogeneous
        gps_lidar_t = tf.quaternion_matrix(gps_lidar_rotation)
        gps_lidar_t[:, 3] = gps_lidar_translation

        # lidar->gps 
        lidar_gps_t = homogeneous_inverse(gps_lidar_t)
        gps_obj_t = gps_ref_t@ref_obj_t

        # get lidar->obj transform
        lidar_obj_t = lidar_gps_t@gps_obj_t

        obj_rel_position = tf.translation_from_matrix(lidar_obj_t)
        obj_rel_orientation_rpy = tf.euler_from_matrix(lidar_obj_t)
        obj_x_rel = obj_rel_position[0]
        obj_y_rel = obj_rel_position[1]
        # ego_obj_t = ego_ref_t * ref_obj_t
        # ego_obj_rpy = tf.euler_from_matrix(ego_obj_t)

        ego_yaw = ref_gps_rpy[2]
        print(f"ego rpy {ref_gps_rpy}")
        print(f"ego yaw degrees:  {np.degrees(ego_yaw)}")
        print(f"obj yaw rel to ref: {np.degrees(ref_obj_yaw)}")
        print(f"obj yaw rel to ego: {np.degrees(obj_rel_orientation_rpy[2])}")

        
        # obj_x_abs, obj_y_abs, _, _ = utm.from_latlon(object_status.latitude, object_status.longitude)
        # obj_x_rel, obj_y_rel = obj_x_ref-ego_x, obj_y_ref-ego_y

        # dist = math.sqrt((obj_x_rel)**2+(obj_y_rel)**2)

        cloud = point_cloud(np.array([[obj_x_rel, obj_y_rel, 0]]), "base_link", "xyz")
        gt_bboxes = BoundingBoxArray()
        bbox = BoundingBox()
        centroid = Point()
        dimension = Point()
        orientation = Quaternion()

        centroid.x = obj_x_rel
        centroid.y = obj_y_rel
        centroid.z = -1.
        dimension.x = 4.
        dimension.y = 2.
        dimension.z = 2.

        # orientation_tf = tf.quaternion_from_matrix(lidar_obj_t) # # obj_rel_orientation_rpy[0]) #+np.deg2rad(90))
        orientation_tf =  tf.quaternion_from_euler(0,0, obj_rel_orientation_rpy[2]+np.deg2rad(90))
        orientation.x = orientation_tf[0]
        orientation.y = orientation_tf[1]
        orientation.z = orientation_tf[2]
        orientation.w = orientation_tf[3]

        bbox.centroid = centroid
        bbox.dimension = dimension
        bbox.orientation = orientation
        bbox.tracking_id = -1
        
        gt_bboxes.bounding_boxes.append(bbox)
        gt_bboxes.header.frame_id = ego_gps_location.header.frame_id
        gt_bboxes.header.stamp = ego_gps_location.header.stamp

        print(f"object's location: ({obj_x_rel}, {obj_y_rel}")
        self.gt_bboxes_publisher_.publish(gt_bboxes)
        # self.pred_bboxes_publisher_.publish(pred_bboxes)
        self.object_1_pc_publisher_.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    tracking_gt = TrackingGT()
    rclpy.spin(tracking_gt)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
