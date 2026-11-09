import { MasterDashboard } from "@/components/master-dashboard";

export const metadata = {
  title: "ARGUS · Dashboard",
  description: "Live globe of the SciVis 2026 climate, atmosphere, and ocean datasets.",
};

export default function DashboardPage() {
  return <MasterDashboard />;
}
