export type Profile = {
  id: string;
  name: string;
  createdAt: string;
};

export type Message = {
  id: string;
  role: "user" | "agent";
  content: string;
  timestamp: string;
};

export type ProfileFile = {
  profile: Profile;
  messages: Message[];
};
