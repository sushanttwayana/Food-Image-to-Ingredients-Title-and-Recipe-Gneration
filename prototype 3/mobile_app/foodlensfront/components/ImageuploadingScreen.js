import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  SafeAreaView,
  Alert,
  ScrollView,
  ActivityIndicator,
  Modal,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import Ionicons from "react-native-vector-icons/Ionicons";
import { useNavigation } from "@react-navigation/native";

const ImageUploadScreen = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recipeData, setRecipeData] = useState(null);
  const [showRecipeModal, setShowRecipeModal] = useState(false);
  const navigation = useNavigation();

  const handleHome = () => navigation.navigate("FoodLens");
  const handlePerson = () => navigation.navigate("Profile");

  const handleImageUpload = () => {
    Alert.alert(
      "Select Image",
      "Choose an option",
      [
        { text: "Camera", onPress: openCamera },
        { text: "Gallery", onPress: openGallery },
        { text: "Cancel", style: "cancel" },
      ],
      { cancelable: true }
    );
  };

  const openCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Denied", "Camera permission is required.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled && result.assets && result.assets[0]) {
      const asset = result.assets[0];
      setSelectedImage({
        uri: asset.uri,
        type: "image/jpeg",
        fileName: `camera_image_${Date.now()}.jpg`,
        fileSize: asset.fileSize,
        width: asset.width,
        height: asset.height,
      });
      setRecipeData(null);
    }
  };

  const openGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Denied", "Gallery permission is required.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (!result.canceled && result.assets && result.assets[0]) {
      const asset = result.assets[0];
      const fileType = asset.uri.split(".").pop();

      setSelectedImage({
        uri: asset.uri,
        type: `image/${fileType}`,
        fileName: `gallery_image_${Date.now()}.${fileType}`,
        fileSize: asset.fileSize,
        width: asset.width,
        height: asset.height,
      });
      setRecipeData(null);
    }
  };

  const uploadImageToAPI = async (image) => {
    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("image", {
        uri: image.uri,
        type: image.type || "image/jpeg",
        name: image.fileName || "image.jpg",
      });

      const res = await fetch("http://192.168.80.148:8000/api/predict/", {
        method: "POST",
        headers: { "Content-Type": "multipart/form-data" },
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      return data;
    } catch (err) {
      console.error("Upload Error:", err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const handleViewRecipe = async () => {
    if (!selectedImage) {
      Alert.alert("No Image", "Please upload an image first");
      return;
    }

    try {
      const data = await uploadImageToAPI(selectedImage);
      setRecipeData(data);
      setShowRecipeModal(true);
    } catch (err) {
      Alert.alert("Error", "Failed to fetch recipe: " + err.message);
    }
  };

  const RecipeModal = () => (
    <Modal
      visible={showRecipeModal}
      animationType="slide"
      presentationStyle="pageSheet"
    >
      <SafeAreaView style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <TouchableOpacity
            onPress={() => setShowRecipeModal(false)}
            style={styles.closeButton}
          >
            <Ionicons name="close" size={24} color="#333" />
          </TouchableOpacity>
          <Text style={styles.modalTitle}>Recipe Details</Text>
        </View>
        <ScrollView style={styles.modalContent}>
          {recipeData && (
            <>
              <Text style={styles.recipeSectionTitle}>Dish Name</Text>
              <Text style={styles.recipeTitle}>
                {recipeData.title || recipeData.dish_name || "Unknown Dish"}
              </Text>

              <Text style={styles.recipeSectionTitle}>Ingredients</Text>
              {(recipeData.ingredients || []).map((ing, i) => (
                <Text key={i} style={styles.ingredientText}>
                  • {ing}
                </Text>
              ))}

              <Text style={styles.recipeSectionTitle}>Instructions</Text>
              <Text style={styles.instructionText}>
                {recipeData.instructions ||
                  recipeData.recipe ||
                  "No instructions available"}
              </Text>
            </>
          )}
        </ScrollView>
      </SafeAreaView>
    </Modal>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.content}>
        <View style={styles.header}>
          <Image source={require("../assets/logo.png")} style={styles.logo} />
        </View>

        <View style={styles.titleSection}>
          <View style={styles.plateIcon}>
            <View style={styles.plateInner} />
          </View>
          <Text style={styles.headerText}>Upload Your Image</Text>
        </View>

        <TouchableOpacity style={styles.uploadArea} onPress={handleImageUpload}>
          {selectedImage ? (
            <Image
              source={{ uri: selectedImage.uri }}
              style={styles.uploadedImage}
            />
          ) : (
            <View style={styles.placeholderContent}>
              <Image
                source={require("../assets/newafood.png")}
                style={styles.backgroundImage}
              />
              <Text style={styles.uploadPromptText}>Tap to select image</Text>
            </View>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.uploadButton}
          onPress={handleImageUpload}
        >
          <Text style={styles.uploadButtonText}>
            {selectedImage ? "Change Image" : "Upload your Image Here"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.recipeButton,
            selectedImage && styles.recipeButtonActive,
          ]}
          onPress={handleViewRecipe}
          disabled={isLoading || !selectedImage}
        >
          {isLoading ? (
            <ActivityIndicator size="small" color="#D85A47" />
          ) : (
            <Text style={styles.recipeButtonText}>
              View Recipe & Ingredients
            </Text>
          )}
        </TouchableOpacity>
      </ScrollView>

      <View style={styles.bottomNav}>
        <TouchableOpacity onPress={handleHome} style={styles.navItem}>
          <Ionicons name="home" size={24} color="#FFF" />
        </TouchableOpacity>
        <TouchableOpacity style={[styles.navItem, styles.activeNavItem]}>
          <Ionicons name="restaurant" size={24} color="#D85A47" />
        </TouchableOpacity>
        <TouchableOpacity onPress={handlePerson} style={styles.navItem}>
          <Ionicons name="person" size={24} color="#FFF" />
        </TouchableOpacity>
      </View>

      <RecipeModal />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FFF" },
  content: { padding: 20, paddingTop: 60 },
  header: { alignItems: "center", marginBottom: 20 },
  logo: { width: 80, height: 80, resizeMode: "contain" },
  titleSection: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 30,
  },
  plateIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: "#D85A47",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 15,
  },
  plateInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#D85A47",
  },
  headerText: { fontSize: 18, fontWeight: "600", color: "#333" },
  uploadArea: {
    height: 200,
    backgroundColor: "#E8E5E0",
    borderRadius: 15,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 20,
  },
  uploadedImage: { width: "100%", height: "100%", borderRadius: 15 },
  backgroundImage: {
    position: "absolute",
    width: "100%",
    height: "100%",
    resizeMode: "cover",
    opacity: 0.2,
  },
  placeholderContent: {
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    width: "100%",
  },
  uploadPromptText: { color: "#D85A47", fontSize: 16, fontWeight: "500" },
  uploadButton: {
    backgroundColor: "#D85A47",
    padding: 16,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 20,
  },
  uploadButtonText: { color: "#FFF", fontSize: 16, fontWeight: "600" },
  recipeButton: {
    borderColor: "#999",
    borderWidth: 1,
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
    backgroundColor: "transparent",
    opacity: 0.6,
  },
  recipeButtonActive: {
    borderColor: "#D85A47",
    backgroundColor: "rgba(216, 90, 71, 0.05)",
    opacity: 1,
  },
  recipeButtonText: { color: "#666", fontSize: 16, fontWeight: "500" },
  bottomNav: {
    flexDirection: "row",
    backgroundColor: "#2A2A2A",
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  navItem: { flex: 1, alignItems: "center", paddingVertical: 10 },
  activeNavItem: {
    backgroundColor: "rgba(216, 90, 71, 0.15)",
    borderRadius: 8,
  },
  modalContainer: { flex: 1, backgroundColor: "#FFF" },
  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 15,
    borderBottomColor: "#E5E5E5",
    borderBottomWidth: 1,
    position: "relative",
  },
  closeButton: { position: "absolute", left: 20 },
  modalTitle: { fontSize: 18, fontWeight: "600", color: "#333" },
  modalContent: { padding: 20 },
  recipeSectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#D85A47",
    marginTop: 20,
  },
  recipeTitle: { fontSize: 24, fontWeight: "600", color: "#333" },
  ingredientText: { fontSize: 16, color: "#555", marginVertical: 2 },
  instructionText: {
    fontSize: 16,
    color: "#555",
    marginTop: 10,
    lineHeight: 22,
  },
});

export default ImageUploadScreen;

// import React, { useState } from "react";
// import {
//   View,
//   Text,
//   StyleSheet,
//   TouchableOpacity,
//   Image,
//   SafeAreaView,
//   Alert,
//   ScrollView,
//   ActivityIndicator,
//   Modal,
// } from "react-native";
// import * as ImagePicker from "expo-image-picker";
// import Ionicons from "react-native-vector-icons/Ionicons";
// import { useNavigation } from "@react-navigation/native";

// const ImageUploadScreen = () => {
//   const [selectedImage, setSelectedImage] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);
//   const [recipeData, setRecipeData] = useState(null);
//   const [showRecipeModal, setShowRecipeModal] = useState(false);
//   const navigation = useNavigation();

//   const handleImageUpload = () => {
//     Alert.alert(
//       "Select Image",
//       "Choose an option",
//       [
//         {
//           text: "Camera",
//           onPress: () => openCamera(),
//         },
//         {
//           text: "Gallery",
//           onPress: () => openGallery(),
//         },
//         {
//           text: "Cancel",
//           style: "cancel",
//         },
//       ],
//       { cancelable: true }
//     );
//   };

//   const openCamera = async () => {
//     try {
//       // Request camera permissions
//       const cameraPermission =
//         await ImagePicker.requestCameraPermissionsAsync();

//       if (cameraPermission.status !== "granted") {
//         Alert.alert(
//           "Permission Denied",
//           "Camera permission is required to take photos"
//         );
//         return;
//       }

//       const result = await ImagePicker.launchCameraAsync({
//         mediaTypes: ImagePicker.MediaTypeOptions.Images,
//         allowsEditing: true,
//         aspect: [4, 3],
//         quality: 0.8,
//         exif: false,
//       });

//       if (!result.canceled && result.assets && result.assets[0]) {
//         const imageAsset = result.assets[0];
//         console.log("Camera Image selected:", imageAsset);

//         setSelectedImage({
//           uri: imageAsset.uri,
//           type: "image/jpeg",
//           fileName: `camera_image_${Date.now()}.jpg`,
//           fileSize: imageAsset.fileSize,
//           width: imageAsset.width,
//           height: imageAsset.height,
//         });
//         setRecipeData(null);
//       }
//     } catch (error) {
//       console.log("Camera Error:", error);
//       Alert.alert("Error", "Failed to take photo: " + error.message);
//     }
//   };

//   const openGallery = async () => {
//     try {
//       // Request media library permissions
//       const mediaPermission =
//         await ImagePicker.requestMediaLibraryPermissionsAsync();

//       if (mediaPermission.status !== "granted") {
//         Alert.alert(
//           "Permission Denied",
//           "Gallery permission is required to select photos"
//         );
//         return;
//       }

//       const result = await ImagePicker.launchImageLibraryAsync({
//         mediaTypes: ImagePicker.MediaTypeOptions.Images,
//         allowsEditing: true,
//         aspect: [4, 3],
//         quality: 0.8,
//         exif: false,
//       });

//       if (!result.canceled && result.assets && result.assets[0]) {
//         const imageAsset = result.assets[0];
//         console.log("Gallery Image selected:", imageAsset);

//         // Get file extension from URI
//         const uriParts = imageAsset.uri.split(".");
//         const fileType = uriParts[uriParts.length - 1];

//         setSelectedImage({
//           uri: imageAsset.uri,
//           type: `image/${fileType}`,
//           fileName: `gallery_image_${Date.now()}.${fileType}`,
//           fileSize: imageAsset.fileSize,
//           width: imageAsset.width,
//           height: imageAsset.height,
//         });
//         setRecipeData(null);
//       }
//     } catch (error) {
//       console.log("Gallery Error:", error);
//       Alert.alert("Error", "Failed to select image: " + error.message);
//     }
//   };

//   const uploadImageToAPI = async (imageAsset) => {
//     try {
//       setIsLoading(true);

//       const formData = new FormData();
//       formData.append("image", {
//         uri: imageAsset.uri,
//         type: imageAsset.type || "image/jpeg",
//         name: imageAsset.fileName || "image.jpg",
//       });

//       console.log("Uploading image to API...");
//       console.log("Image details:", {
//         uri: imageAsset.uri,
//         type: imageAsset.type,
//         name: imageAsset.fileName,
//       });

//       const response = await fetch("http://192.168.1.69:8000/api/predict/", {
//         method: "POST",
//         body: formData,
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//       });

//       console.log("API Response status:", response.status);

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();
//       console.log("API Response data:", data);
//       return data;
//     } catch (error) {
//       console.error("Error uploading image:", error);
//       throw error;
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleViewRecipe = async () => {
//     if (!selectedImage) {
//       Alert.alert("No Image", "Please upload an image first");
//       return;
//     }

//     try {
//       const recipe = await uploadImageToAPI(selectedImage);
//       setRecipeData(recipe);
//       setShowRecipeModal(true);
//     } catch (error) {
//       Alert.alert(
//         "Error",
//         "Failed to get recipe. Please check your connection and try again.\n\nError: " +
//           error.message
//       );
//     }
//   };

//   const RecipeModal = () => (
//     <Modal
//       visible={showRecipeModal}
//       animationType="slide"
//       presentationStyle="pageSheet"
//     >
//       <SafeAreaView style={styles.modalContainer}>
//         <View style={styles.modalHeader}>
//           <TouchableOpacity
//             style={styles.closeButton}
//             onPress={() => setShowRecipeModal(false)}
//           >
//             <Ionicons name="close" size={24} color="#333" />
//           </TouchableOpacity>
//           <Text style={styles.modalTitle}>Recipe Details</Text>
//         </View>

//         <ScrollView style={styles.modalContent}>
//           {recipeData && (
//             <>
//               <View style={styles.recipeSection}>
//                 <Text style={styles.recipeSectionTitle}>Dish Name</Text>
//                 <Text style={styles.recipeTitle}>
//                   {recipeData.title || recipeData.dish_name || "Unknown Dish"}
//                 </Text>
//               </View>

//               <View style={styles.recipeSection}>
//                 <Text style={styles.recipeSectionTitle}>Ingredients</Text>
//                 {(recipeData.ingredients || []).map((ingredient, index) => (
//                   <View key={index} style={styles.ingredientItem}>
//                     <Text style={styles.ingredientBullet}>•</Text>
//                     <Text style={styles.ingredientText}>{ingredient}</Text>
//                   </View>
//                 ))}
//                 {(!recipeData.ingredients ||
//                   recipeData.ingredients.length === 0) && (
//                   <Text style={styles.noDataText}>
//                     No ingredients available
//                   </Text>
//                 )}
//               </View>

//               <View style={styles.recipeSection}>
//                 <Text style={styles.recipeSectionTitle}>Instructions</Text>
//                 <Text style={styles.instructionText}>
//                   {recipeData.instructions ||
//                     recipeData.recipe ||
//                     "No instructions available"}
//                 </Text>
//               </View>
//             </>
//           )}
//           {!recipeData && (
//             <View style={styles.noDataContainer}>
//               <Text style={styles.noDataText}>No recipe data available</Text>
//             </View>
//           )}
//         </ScrollView>
//       </SafeAreaView>
//     </Modal>
//   );

//   return (
//     <SafeAreaView style={styles.container}>
//       <ScrollView style={styles.content}>
//         {/* Header */}
//         <View style={styles.header}>
//           <View style={styles.logoContainer}>
//             <Image source={require("../assets/logo.png")} style={styles.logo} />
//           </View>
//         </View>

//         {/* Title Section */}
//         <View style={styles.titleSection}>
//           <View style={styles.iconContainer}>
//             <View style={styles.plateIcon}>
//               <View style={styles.plateInner} />
//             </View>
//           </View>
//           <Text style={styles.headerText}>Upload Your Image</Text>
//         </View>

//         {/* Upload Area */}
//         <TouchableOpacity
//           style={styles.uploadArea}
//           onPress={handleImageUpload}
//           activeOpacity={0.7}
//         >
//           {selectedImage ? (
//             <Image
//               source={{ uri: selectedImage.uri }}
//               style={styles.uploadedImage}
//             />
//           ) : (
//             <View style={styles.placeholderContent}>
//               <Image
//                 source={require("../assets/newafood.png")}
//                 style={styles.backgroundImage}
//               />
//               <View style={styles.overlayPattern}>
//                 <View style={styles.patternOverlay} />
//               </View>
//               <View style={styles.uploadPrompt}>
//                 <Ionicons name="camera-outline" size={40} color="#D85A47" />
//                 <Text style={styles.uploadPromptText}>Tap to select image</Text>
//               </View>
//             </View>
//           )}
//         </TouchableOpacity>

//         {/* Upload Button */}
//         <TouchableOpacity
//           style={styles.uploadButton}
//           onPress={handleImageUpload}
//           activeOpacity={0.8}
//         >
//           <Text style={styles.uploadButtonIcon}>⬆</Text>
//           <Text style={styles.uploadButtonText}>
//             {selectedImage ? "Change Image" : "Upload your Image Here"}
//           </Text>
//         </TouchableOpacity>

//         {/* View Recipe Button */}
//         <TouchableOpacity
//           style={[
//             styles.recipeButton,
//             selectedImage && styles.recipeButtonActive,
//           ]}
//           onPress={handleViewRecipe}
//           activeOpacity={0.8}
//           disabled={isLoading || !selectedImage}
//         >
//           {isLoading ? (
//             <View style={styles.loadingContainer}>
//               <ActivityIndicator
//                 size="small"
//                 color={selectedImage ? "#D85A47" : "#666"}
//               />
//               <Text
//                 style={[
//                   styles.recipeButtonText,
//                   selectedImage && styles.recipeButtonTextActive,
//                   { marginLeft: 10 },
//                 ]}
//               >
//                 Processing...
//               </Text>
//             </View>
//           ) : (
//             <>
//               <Text
//                 style={[
//                   styles.recipeButtonText,
//                   selectedImage && styles.recipeButtonTextActive,
//                 ]}
//               >
//                 View Recipe & Ingredients
//               </Text>
//               <Text
//                 style={[
//                   styles.recipeButtonArrow,
//                   selectedImage && styles.recipeButtonArrowActive,
//                 ]}
//               >
//                 →
//               </Text>
//             </>
//           )}
//         </TouchableOpacity>

//         {/* Debug Info - Remove in production */}
//         {selectedImage && (
//           <View style={styles.debugContainer}>
//             <Text style={styles.debugTitle}>Selected Image Info:</Text>
//             {/* <Text style={styles.debugText}>URI: {selectedImage.uri}</Text> */}
//             <Text style={styles.debugText}>Type: {selectedImage.type}</Text>
//             <Text style={styles.debugText}>
//               Size:{" "}
//               {selectedImage.fileSize
//                 ? `${Math.round(selectedImage.fileSize / 1024)} KB`
//                 : "Unknown"}
//             </Text>
//             <Text style={styles.debugText}>
//               Dimensions: {selectedImage.width}x{selectedImage.height}
//             </Text>
//           </View>
//         )}
//       </ScrollView>

//       {/* Bottom Navigation */}
//       <View style={styles.bottomNav}>
//         <TouchableOpacity style={styles.navItem}>
//           <Ionicons name="home" size={24} color="#999" />
//         </TouchableOpacity>
//         <TouchableOpacity style={[styles.navItem, styles.activeNavItem]}>
//           <Ionicons name="restaurant" size={24} color="#D85A47" />
//         </TouchableOpacity>
//         <TouchableOpacity style={styles.navItem}>
//           <Ionicons name="search" size={24} color="#999" />
//         </TouchableOpacity>
//         <TouchableOpacity style={styles.navItem}>
//           <Ionicons name="person" size={24} color="#999" />
//         </TouchableOpacity>
//       </View>

//       <RecipeModal />
//     </SafeAreaView>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: "#FFFFFF",
//   },
//   content: {
//     flex: 1,
//     paddingHorizontal: 20,
//     paddingTop: 60,
//   },
//   header: {
//     alignItems: "center",
//     marginBottom: 20,
//   },
//   logoContainer: {
//     alignItems: "center",
//     justifyContent: "center",
//   },
//   logo: {
//     width: 80,
//     height: 80,
//     resizeMode: "contain",
//   },
//   titleSection: {
//     flexDirection: "row",
//     alignItems: "center",
//     marginBottom: 30,
//     marginTop: 30,
//   },
//   iconContainer: {
//     marginRight: 15,
//   },
//   plateIcon: {
//     width: 30,
//     height: 30,
//     borderRadius: 15,
//     borderWidth: 2,
//     borderColor: "#D85A47",
//     justifyContent: "center",
//     alignItems: "center",
//   },
//   plateInner: {
//     width: 12,
//     height: 12,
//     borderRadius: 6,
//     backgroundColor: "#D85A47",
//   },
//   headerText: {
//     fontSize: 18,
//     fontWeight: "600",
//     color: "#333",
//   },
//   uploadArea: {
//     height: 200,
//     borderRadius: 15,
//     backgroundColor: "#E8E5E0",
//     marginBottom: 30,
//     overflow: "hidden",
//     position: "relative",
//   },
//   uploadedImage: {
//     width: "100%",
//     height: "100%",
//     borderRadius: 15,
//   },
//   placeholderContent: {
//     flex: 1,
//     position: "relative",
//     justifyContent: "center",
//     alignItems: "center",
//   },
//   backgroundImage: {
//     position: "absolute",
//     width: "100%",
//     height: "100%",
//     resizeMode: "cover",
//     opacity: 0.2,
//   },
//   overlayPattern: {
//     position: "absolute",
//     width: "100%",
//     height: "100%",
//     backgroundColor: "rgba(248, 245, 240, 0.3)",
//   },
//   patternOverlay: {
//     flex: 1,
//     backgroundColor: "transparent",
//   },
//   uploadPrompt: {
//     alignItems: "center",
//     justifyContent: "center",
//     zIndex: 1,
//   },
//   uploadPromptText: {
//     marginTop: 10,
//     fontSize: 16,
//     color: "#D85A47",
//     fontWeight: "500",
//   },
//   uploadButton: {
//     backgroundColor: "#D85A47",
//     borderRadius: 12,
//     paddingVertical: 16,
//     flexDirection: "row",
//     alignItems: "center",
//     justifyContent: "center",
//     marginBottom: 20,
//   },
//   uploadButtonIcon: {
//     color: "white",
//     fontSize: 18,
//     marginRight: 10,
//     fontWeight: "bold",
//   },
//   uploadButtonText: {
//     color: "white",
//     fontSize: 16,
//     fontWeight: "600",
//   },
//   recipeButton: {
//     backgroundColor: "transparent",
//     borderWidth: 1,
//     borderColor: "#999",
//     borderRadius: 12,
//     paddingVertical: 16,
//     flexDirection: "row",
//     alignItems: "center",
//     justifyContent: "center",
//     opacity: 0.6,
//   },
//   recipeButtonActive: {
//     borderColor: "#D85A47",
//     backgroundColor: "rgba(216, 90, 71, 0.05)",
//     opacity: 1,
//   },
//   recipeButtonText: {
//     color: "#666",
//     fontSize: 16,
//     fontWeight: "500",
//     marginRight: 10,
//   },
//   recipeButtonTextActive: {
//     color: "#D85A47",
//   },
//   recipeButtonArrow: {
//     color: "#666",
//     fontSize: 16,
//   },
//   recipeButtonArrowActive: {
//     color: "#D85A47",
//   },
//   loadingContainer: {
//     flexDirection: "row",
//     alignItems: "center",
//   },
//   bottomNav: {
//     flexDirection: "row",
//     backgroundColor: "#2A2A2A",
//     paddingVertical: 15,
//     paddingHorizontal: 20,
//     borderTopLeftRadius: 20,
//     borderTopRightRadius: 20,
//   },
//   navItem: {
//     flex: 1,
//     alignItems: "center",
//     paddingVertical: 10,
//   },
//   activeNavItem: {
//     backgroundColor: "rgba(216, 90, 71, 0.15)",
//     borderRadius: 8,
//   },
//   // Modal Styles
//   modalContainer: {
//     flex: 1,
//     backgroundColor: "#FFFFFF",
//   },
//   modalHeader: {
//     flexDirection: "row",
//     alignItems: "center",
//     justifyContent: "center",
//     paddingHorizontal: 20,
//     paddingVertical: 15,
//     borderBottomWidth: 1,
//     borderBottomColor: "#E5E5E5",
//     position: "relative",
//   },
//   closeButton: {
//     position: "absolute",
//     left: 20,
//     padding: 5,
//   },
//   modalTitle: {
//     fontSize: 18,
//     fontWeight: "600",
//     color: "#333",
//   },
//   modalContent: {
//     flex: 1,
//     paddingHorizontal: 20,
//     paddingTop: 20,
//   },
//   recipeSection: {
//     marginBottom: 25,
//   },
//   recipeSectionTitle: {
//     fontSize: 16,
//     fontWeight: "700",
//     color: "#D85A47",
//     marginBottom: 10,
//     textTransform: "uppercase",
//     letterSpacing: 0.5,
//   },
//   recipeTitle: {
//     fontSize: 24,
//     fontWeight: "600",
//     color: "#333",
//     textTransform: "capitalize",
//   },
//   ingredientItem: {
//     flexDirection: "row",
//     alignItems: "flex-start",
//     marginBottom: 8,
//   },
//   ingredientBullet: {
//     fontSize: 16,
//     color: "#D85A47",
//     marginRight: 10,
//     marginTop: 2,
//   },
//   ingredientText: {
//     fontSize: 16,
//     color: "#555",
//     flex: 1,
//     textTransform: "capitalize",
//   },
//   instructionText: {
//     fontSize: 16,
//     color: "#555",
//     lineHeight: 24,
//     textAlign: "justify",
//   },
//   noDataContainer: {
//     alignItems: "center",
//     justifyContent: "center",
//     paddingVertical: 40,
//   },
//   noDataText: {
//     fontSize: 16,
//     color: "#999",
//     fontStyle: "italic",
//   },
//   // Debug styles - Remove in production
//   debugContainer: {
//     marginTop: 20,
//     padding: 15,
//     backgroundColor: "#f0f0f0",
//     borderRadius: 8,
//   },
//   debugTitle: {
//     fontSize: 14,
//     fontWeight: "600",
//     color: "#333",
//     marginBottom: 5,
//   },
//   debugText: {
//     fontSize: 12,
//     color: "#666",
//     marginBottom: 3,
//   },
// });

// export default ImageUploadScreen;
