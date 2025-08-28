import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";

const ProfileSettingsScreen = () => {
  const navigation = useNavigation(); // Move this inside the component

  const handleHome = () => {
    navigation.navigate("FoodLens");
  };

  const handleUpload = () => {
    navigation.navigate("ImageUploading");
  };

  const handlePerson = () => {
    navigation.navigate("Profile");
  };

  const handleLogout = () => {
    navigation.navigate("startScreen");
  };

  const menuItems = [
    {
      id: 1,
      title: "Edit Profile",
      icon: "person-outline",
      onPress: () => console.log("Edit Profile pressed"),
    },
    {
      id: 2,
      title: "Update Password",
      icon: "lock-closed-outline",
      onPress: () => console.log("Update Password pressed"),
    },
    {
      id: 3,
      title: "Contact Support",
      icon: "call-outline",
      onPress: () => console.log("Contact Support pressed"),
    },
    {
      id: 4,
      title: "Log Out",
      icon: "log-out-outline",
      onPress: handleLogout, // Use the handleLogout function
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F5F5DC" />

      <View style={styles.content}>
        {/* Header with Icon and Greeting */}
        <View style={styles.header}>
          <View style={styles.headerIcon}>
            <Image
              source={require("../assets/logo.png")}
              style={styles.logoImage}
              resizeMode="contain"
            />
          </View>
          <Text style={styles.greeting}>FoodLens App</Text>
        </View>

        {/* Profile Image */}
        <View style={styles.profileSection}>
          <View style={styles.profileImageContainer}>
            <Image
              source={require("../assets/burger.png")}
              style={styles.profileImage}
            />
            <View style={styles.editIcon}>
              <Ionicons name="pencil" size={16} color="white" />
            </View>
          </View>
        </View>

        {/* Menu Card */}
        <View style={styles.menuCard}>
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={item.id}
              style={[
                styles.menuItem,
                index === menuItems.length - 1 && styles.lastMenuItem,
              ]}
              onPress={item.onPress}
            >
              <View style={styles.menuItemContent}>
                <Ionicons name={item.icon} size={20} color="white" />
                <Text style={styles.menuItemText}>{item.title}</Text>
              </View>
            </TouchableOpacity>
          ))}

          {/* App Logo at bottom of card */}
          <View style={styles.appLogoSection}>
            <View style={styles.appLogo}>
              <Ionicons name="close" size={16} color="#C4342C" />
            </View>
            <Text style={styles.appName}>FoodLens</Text>
          </View>

          {/* Project Footer */}
          <View style={styles.projectFooter}>
            <Text style={styles.projectTitle}>Project Done By:</Text>
            <View style={styles.developersContainer}>
              <Text style={styles.developerName}>Jenish Prajapati</Text>
              <Text style={styles.developerName}>Roji Prajapati</Text>
              <Text style={styles.developerName}>Shreeya Shrestha</Text>
              <Text style={styles.developerName}>Sumina Awa</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity onPress={handleHome} style={styles.navItem}>
          <Ionicons name="home" size={24} color="#FFF" />
        </TouchableOpacity>
        <TouchableOpacity onPress={handleUpload} style={styles.navItem}>
          <Ionicons name="restaurant" size={24} color="#FFF" />
        </TouchableOpacity>
        <TouchableOpacity onPress={handlePerson} style={styles.navItem}>
          <Ionicons name="person" size={24} color="#D85A47" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F5DC",
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingTop: 20,
    paddingBottom: 30,
  },
  headerIcon: {
    width: 80,
    height: 80,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 15,
  },
  logoImage: {
    width: 70,
    height: 70,
  },
  greeting: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
  },
  profileSection: {
    alignItems: "center",
    marginBottom: 30,
  },
  profileImageContainer: {
    position: "relative",
  },
  profileImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 4,
    borderColor: "white",
  },
  editIcon: {
    position: "absolute",
    bottom: 5,
    right: 5,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#C4342C",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "white",
  },
  menuCard: {
    backgroundColor: "white",
    borderRadius: 15,
    paddingVertical: 10,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
  },
  menuItem: {
    backgroundColor: "#C4342C",
    marginHorizontal: 15,
    marginVertical: 5,
    borderRadius: 8,
    paddingVertical: 15,
    paddingHorizontal: 20,
  },
  lastMenuItem: {
    marginBottom: 20,
  },
  menuItemContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  menuItemText: {
    color: "white",
    fontSize: 16,
    fontWeight: "500",
    marginLeft: 15,
  },
  appLogoSection: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingBottom: 15,
  },
  appLogo: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "white",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#C4342C",
    marginRight: 8,
  },
  appName: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
  },
  projectFooter: {
    borderTopWidth: 1,
    borderTopColor: "#E5E5E5",
    paddingTop: 15,
    paddingBottom: 15,
    paddingHorizontal: 15,
    alignItems: "center",
  },
  projectTitle: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#666",
    marginBottom: 8,
  },
  developersContainer: {
    alignItems: "center",
  },
  developerName: {
    fontSize: 13,
    color: "#888",
    marginBottom: 2,
    fontWeight: "500",
  },
  bottomNav: {
    flexDirection: "row",
    backgroundColor: "#2C2C2C",
    paddingVertical: 15,
    paddingHorizontal: 20,
    justifyContent: "space-around",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  navItem: {
    padding: 5,
  },
});

export default ProfileSettingsScreen;
