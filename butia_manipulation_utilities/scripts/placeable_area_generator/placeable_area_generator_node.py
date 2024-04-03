#!/usr/bin/env python3
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import rospy
from butia_vision_msgs.srv import PlaceableArea, PlaceableAreaResponse
import open3d as o3d
from scipy.spatial import ConvexHull
import cv2

class PlaceableAreaNode:

    def _init_(self, init_node=True):
        self.initRosComm()

    def callback(self, data: PlaceableArea):
        print('opaaa')
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
        print('beleza?')

        cloud = data.cloud # Read cloud with o3d.io.read_point_cloud

        # Segment the plane
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
        [a, b, c, d] = plane_model # a*x + b*y + c*z + d = 0

        inlier_cloud = cloud.select_by_index(inliers)
        outlier_cloud = cloud.select_by_index(inliers, invert=True)

        # Get the points above the plane
        above_plane_indices = np.where(a * np.array(outlier_cloud.points)[:, 0] + b * np.array(outlier_cloud.points)[:, 1] + c * np.array(outlier_cloud.points)[:, 2] + d > 0)[0]
        above_plane_cloud = outlier_cloud.select_by_index(above_plane_indices)

        above_plane_cloud_projected = self.get_projected_cloud(plane_model, above_plane_cloud)

        # Clusterize the points above the plane
        cloud = self.clustering(above_plane_cloud_projected)

        # Change this to get another way to extract the object
        obj_cloud = self.extract_obj(cloud)

        hull = self.get_hull(obj_cloud)

        obj_2d = self.to2d(plane_model, hull)

        hull2d = self.get_hull2d(obj_2d)

        hull_points = obj_2d[hull2d]
        
        hull_points = np.concatenate([hull_points, hull_points[0:1]]) # Close the polygon

        xy_hull = self.get_xy_by_hull(obj_2d, hull2d)
        
        # TODO: Get the plane orientation and limits
        plane = {'x': [-1, 1, 1, -1], 'y': [-1, -1, 1, 1]}

        # TODO: Get image by an array of hulls for all object in one image
        image = self.from_xy_to_img(xy_hull)
        
        # TODO: Center for every hull of the same label
        hull_center = (np.mean(xy_hull['y']), np.mean(xy_hull['x']))
        hull_center = (int(hull_center[0] * 500 + 500), int(hull_center[1] * 500 + 500)) # Normalizing the center to fit the image

        placeable_point_xy = self.bfs(image, hull_center)

        if placeable_point_xy is None:
            print('No placeable point found')
            response = PlaceableAreaResponse()
            response.score = 0
            response.id = 0
            return response

        im = self.debug_image(image, placeable_point)
        plt.imshow(im, cmap='gray')
        plt.show()

        # Reverter a normalização
        placeable_point = ((placeable_point[0] - 500) / 500, (placeable_point[1] - 500) / 500)

        placeable_point_np = np.array(placeable_point)

        print(placeable_point_np)
        # Adicionar uma terceira coordenada ao vertices
        vertices_3d = np.append(placeable_point_np, 0)  # Adicionar 0 como a coordenada z

        # Agora vertices_3d pode ser multiplicado pela rotation_matrix
        central_placeable = self.to3d(plane_model, vertices_3d)
        print('central_placeable', central_placeable)
        central_coordinates = np.asarray(central_placeable.points)[0]

        # Transformar o ponto central em uma esfera
        radius = .05  # Defina o raio da esfera como desejado
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(central_coordinates)  # Transladar a esfera para o ponto central

        # Converter a esfera (TriangleMesh) em uma PointCloud
        sphere_pcd = sphere.sample_points_uniformly(number_of_points=1000)

        projected_sphere = self.get_projected_cloud(plane_model, sphere_pcd)

        # Visualizar a esfera e outros objetos
        self.visualize_3d([obj_cloud, projected_sphere, sphere, hull])

        response = PlaceableAreaResponse()

        response.x = placeable_point_np[0]
        response.y = placeable_point_np[1]
        response.z = placeable_point_np[2]
        response.score = 1.0
        response.id = 1

        return response
        
    def initRosComm(self):
        self.placeable_area_srv = rospy.Service('placeable_area', PlaceableArea, self.callback)

    def get_projected_cloud(self, plane_model: List[float], cloud: o3d.geometry.PointCloud):
        [a, b, c, d] = plane_model
        projected_points = []
        for point in np.array(cloud.points):
            t = -(a*point[0] + b*point[1] + c*point[2] + d) / (a*a + b*b + c*c)
            projected_point = point + np.array([a, b, c]) * t
            projected_points.append(projected_point)
        projected_cloud = o3d.geometry.PointCloud()
        projected_cloud.points = o3d.utility.Vector3dVector(projected_points)
        return projected_cloud
    
    def clustering(self, cloud: o3d.geometry.PointCloud):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        colors = cloud.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return cloud

    def extract_obj(self, cloud):
        obj_color = np.array([0.83921569, 0.15294118, 0.15686275])

        tolerance = 0.1
        colors = np.asarray(cloud.colors)
        indices = np.where(np.linalg.norm(colors - obj_color, axis=1) < tolerance)[0]

        obj_cloud = cloud.select_by_index(indices)

        # o3d.visualization.draw_geometries([obj_cloud])
        return obj_cloud

    def get_hull(self, cloud):
        hull, _ = cloud.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        return hull_ls
    
    def to2d(self, plane_model: List[float], cloud: o3d.geometry.PointCloud):
        # Calculate the plane orientation
        plane_normal = np.array([plane_model[0], plane_model[1], plane_model[2]])
        plane_orientation = np.arccos(plane_normal.dot([0, 0, 1]))

        # Calculate the rotation matrix to align the plane with the XY plane
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((plane_orientation, 0, 0))

        # Project the points of the point cloud onto the XY plane
        try:
            points = np.asarray(cloud.points)
        except:
            points = np.asarray(cloud)
        points = points.dot(rotation_matrix.T)  # Transpose the rotation matrix to undo the rotation

        return points

    def get_hull2d(self, points: List[List[float]]):
        # Calculate the 2D convex hull
        hull = ConvexHull(points[:, :2])

        # Return the indices of the points that form the convex hull
        return hull.vertices
    
    def get_xy_by_hull(self, points, hull_indices):
        hull_points = points[hull_indices]
        hull_points = np.concatenate([hull_points, hull_points[0:1]])
        edge = {}
        edge['x'] = hull_points[:, 0]
        edge['y'] = hull_points[:, 1]
        return edge

    def from_xy_to_img(self, xy_hull):
        # Criar uma imagem em branco
        image = np.ones((1000, 1000), dtype=np.uint8) * 255

        # Normalizar os pontos do polígono para caber na imagem
        hull_points = np.array([xy_hull['x'], xy_hull['y']]).T * 500 + 500
        hull_points = hull_points.astype(int)

        # Preencher os pontos dentro do polígono com preto
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if self.point_in_hull((x, y), hull_points):
                    image[y, x] = 0

        return image
    
    def point_in_hull(self, point, hull):
        return cv2.pointPolygonTest(hull, point, False) >= 0
    
    def bfs(self, image, hull_center, step_size=20):
        # Initialize queue with the starting point (hull_center)
        queue = [hull_center]

        # Initialize visited matrix with zeros (same size as the image)
        visited = np.zeros(image.shape, dtype=np.uint8)

        while queue:
            # Remove the first point from the queue
            point = queue.pop(0)

            # Check if the point has already been visited
            if visited[point] == 0:
                # Mark the point as visited
                visited[point] = 1

                # Check if the point is placeable
                # TODO: Put the radius as a parameter of the real object
                if self.is_placeable_point(point, image, radius=0.05):
                    return point

                # Add neighbors to the queue (check image boundaries)
                if point[0] > 0: queue.append((point[0]-step_size, point[1]))
                if point[0] < image.shape[0]-1: queue.append((point[0]+step_size, point[1]))
                if point[1] > 0: queue.append((point[0], point[1]-step_size))
                if point[1] < image.shape[1]-1: queue.append((point[0], point[1]+step_size))

        # If breadth-first search doesn't find a placeable point, return None
        return None

    def is_placeable_point(self, point, image, radius):
        if image[point] == 0:
            return False
        # Calculate the radius size in pixels
        radius_pixels = int(radius * 500)

        # Check all points within a square of side 2*radius centered on the point
        for dy in range(-radius_pixels, radius_pixels+1):
            for dx in range(-radius_pixels, radius_pixels+1):
                # Check if the point is inside the image
                if 0 <= point[0]+dy < image.shape[0] and 0 <= point[1]+dx < image.shape[1]:
                    # If the point is black, the circle doesn't fit
                    if image[point[0]+dy, point[1]+dx] == 0:
                        return False

        # If no black point is found, the circle fits
        return True

    def debug_image(self, image, point, radius):
        # Calculate the radius size in pixels
        radius_pixels = int(radius * 500)

        # Create a copy of the image to draw on
        debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

        # Fill the circle with red color
        # Invert the coordinates of the point
        cv2.circle(debug_image, (point[1], point[0]), radius_pixels, (0, 0, 255), -1)

        return debug_image
    
    def to3d(self, plane_model, vertices):
        # Calculate the plane orientation
        plane_normal = np.array([plane_model[0], plane_model[1], plane_model[2]])
        plane_orientation = np.arccos(plane_normal.dot([0, 0, 1]))

        # Calculate the rotation matrix to align the XY plane with the original plane
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -plane_orientation))

        # Transform the vertices back to the 3D world
        vertices = vertices.dot(rotation_matrix.T)  # Apply the rotation
        vertices += plane_model[:3] * plane_model[3]  # Apply the translation

        vertices_list = [vertices]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(vertices_list)

        return cloud

    def visualize_3d(self, pcds):
        o3d.visualization.draw_geometries(pcds)
        

if __name__ == '__main__':
    print('oi1')
    rospy.init_node('placeable_area_node', anonymous = True)
    print('eai1')
    
    # placeable_area = PlaceableAreaNode()
    print('eai')

    rospy.spin()