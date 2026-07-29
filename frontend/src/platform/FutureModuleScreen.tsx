import {
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";

export default function FutureModuleScreen({
  title,
  message,
}: {
  title: string;
  message: string;
}): React.ReactElement {
  return (
    <div className="platform-page">
      <PlatformPageHeader title={title} description="Planned module" />
      <PlatformSection title="Not enabled in this release">
        <div className="platform-planned-state">
          <PlatformStatus value="planned" />
          <p>{message}</p>
        </div>
      </PlatformSection>
    </div>
  );
}
